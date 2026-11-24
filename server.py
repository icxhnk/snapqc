"""SnapQC production & local inference server (v0.4.0):
- Real YOLO11 ONNX detection + optional SAHI 2x2 overlapping tiled micro-defect inference (?tile=1)
- Hybrid open access: anyone can use SnapQC immediately as a guest, OR sign in via SSO (OAuth 2.0 + PKCE S256)
  for private per-user SQLite QC report history
- Fail-closed maintenance mode when both IDENTITY_PLATFORM_ENABLED and PUBLIC_DEMO_ENABLED are false
"""

import base64
import hashlib
import http.cookies
import http.server
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from db import SnapQCStore

try:
    import jwt
    from jwt import PyJWK
except ImportError:  # pragma: no cover
    jwt = None
    PyJWK = None

ROOT = Path(__file__).resolve().parent
VERSION = "0.5.0"
METRICS_V1_PATH = ROOT / "models" / "metrics-v1.json"
METRICS_V2_PATH = ROOT / "models" / "metrics-v2.json"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_JSON_BYTES = 3 * 1024 * 1024

CLASS_NAMES = {0: "Bad Weld", 1: "Good Weld", 2: "Defect"}
SESSION = None
MODEL_INPUT_SIZE = 640
MODEL_DISPLAY_NAME = "SnapQC Welding v1 (YOLO11n ONNX)"
MODEL_FILE_NAME = "snapqc-welding-v1.onnx"
MODEL_LOADING = True
MODEL_ERROR = None
MODEL_READY = threading.Event()

PENDING_OAUTH = {}
DISCOVERY_CACHE = {}
JWKS_CACHE = {}
RATE_BUCKETS = defaultdict(deque)
STATE_LOCK = threading.Lock()
STORE_CACHE: dict[str, SnapQCStore] = {}


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def resolve_default_model_path() -> tuple[Path, str]:
    explicit = os.environ.get("SNAPQC_MODEL_PATH")
    if explicit:
        p = Path(explicit)
        return p, f"SnapQC Custom ({p.name})"
    v2_onnx = ROOT / "models" / "snapqc-welding-v2.onnx"
    if v2_onnx.exists() and METRICS_V2_PATH.exists():
        try:
            m2 = json.loads(METRICS_V2_PATH.read_text(encoding="utf-8"))
            if m2.get("promoted_as_default"):
                return v2_onnx, m2.get("model") or "SnapQC Welding v2 (YOLO11s ONNX)"
        except Exception:
            pass
    v1_onnx = ROOT / "models" / "snapqc-welding-v1.onnx"
    return v1_onnx, "SnapQC Welding v1 (YOLO11n ONNX)"


def get_config() -> dict:
    issuer = (os.environ.get("IDENTITY_ISSUER") or "https://accounts.icxhnk.com").rstrip("/")
    client_id = (os.environ.get("IDENTITY_CLIENT_ID") or "snapqc").strip()
    client_secret = (os.environ.get("IDENTITY_CLIENT_SECRET") or "").strip()
    redirect_uri = (os.environ.get("IDENTITY_REDIRECT_URI") or "").strip()
    # Default empty so any authenticated user granted product access by Identity can sign in
    required_roles_raw = os.environ.get("IDENTITY_REQUIRED_ROLES", "")
    required_roles = {r.strip() for r in required_roles_raw.split(",") if r.strip()}
    identity_enabled = env_bool("IDENTITY_PLATFORM_ENABLED", False)
    public_demo_enabled = env_bool("PUBLIC_DEMO_ENABLED", True)
    app_origin = (os.environ.get("APP_ORIGIN") or "").rstrip("/")
    model_path, model_label = resolve_default_model_path()
    db_path = Path(os.environ.get("SNAPQC_DB_PATH") or (ROOT / "data" / "snapqc.sqlite"))
    rate_limit = max(1, int(os.environ.get("SNAPQC_RATE_LIMIT_PER_MIN", "30")))
    identity_configured = bool(issuer and client_id and client_secret and redirect_uri)
    if identity_enabled and public_demo_enabled:
        auth_mode = "hybrid" if identity_configured else "public-demo"
    elif identity_enabled:
        auth_mode = "identity" if identity_configured else "identity-misconfigured"
    elif public_demo_enabled:
        auth_mode = "public-demo"
    else:
        auth_mode = "maintenance"
    return {
        "host": os.environ.get("HOST", "127.0.0.1"),
        "port": int(os.environ.get("PORT", "57840")),
        "app_origin": app_origin,
        "model_path": model_path,
        "model_label": model_label,
        "db_path": db_path,
        "rate_limit_per_min": rate_limit,
        "identity_enabled": identity_enabled,
        "identity_configured": identity_configured,
        "public_demo_enabled": public_demo_enabled,
        "auth_mode": auth_mode,
        "issuer": issuer,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "required_roles": required_roles,
    }


def get_store(cfg: dict | None = None) -> SnapQCStore:
    cfg = cfg or get_config()
    key = str(cfg["db_path"].resolve())
    with STATE_LOCK:
        store = STORE_CACHE.get(key)
        if store is None:
            store = SnapQCStore(cfg["db_path"])
            STORE_CACHE[key] = store
        return store


def load_model():
    global SESSION, CLASS_NAMES, MODEL_INPUT_SIZE, MODEL_DISPLAY_NAME, MODEL_FILE_NAME, MODEL_LOADING, MODEL_ERROR
    t0 = time.perf_counter()
    cfg = get_config()
    model_path = cfg["model_path"]
    try:
        for mp in (METRICS_V2_PATH, METRICS_V1_PATH):
            if mp.exists():
                meta = json.loads(mp.read_text(encoding="utf-8"))
                if isinstance(meta.get("classes"), dict):
                    CLASS_NAMES = {int(k): v for k, v in meta["classes"].items()}
                break
        if model_path.exists():
            import onnxruntime as ort

            sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            in_shape = sess.get_inputs()[0].shape
            if len(in_shape) >= 4 and isinstance(in_shape[2], int) and in_shape[2] >= 320:
                MODEL_INPUT_SIZE = int(in_shape[2])
            else:
                MODEL_INPUT_SIZE = 640
            SESSION = sess
            MODEL_DISPLAY_NAME = cfg["model_label"]
            MODEL_FILE_NAME = model_path.name
            print(
                f"SnapQC ONNX model ready in {time.perf_counter() - t0:.1f}s "
                f"({MODEL_FILE_NAME}, imgsz={MODEL_INPUT_SIZE})",
                flush=True,
            )
        else:
            MODEL_ERROR = f"Model ONNX tidak ditemukan di {model_path.name}."
    except Exception as exc:
        MODEL_ERROR = str(exc)
        print(f"SnapQC model load failed: {exc}", flush=True)
    finally:
        MODEL_LOADING = False
        MODEL_READY.set()


def letterbox(img, new_shape=640):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (new_shape - nw) / 2, (new_shape - nh) / 2
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, r, left, top


def _infer_single_crop(img_bgr: np.ndarray, conf_thres: float):
    h, w = img_bgr.shape[:2]
    padded, ratio, pad_x, pad_y = letterbox(img_bgr, MODEL_INPUT_SIZE)
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob[None, ...])
    in_name = SESSION.get_inputs()[0].name
    out = SESSION.run(None, {in_name: blob})[0]

    preds = out[0].T
    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]
    cls_ids = np.argmax(class_scores, axis=1)
    confs = class_scores[np.arange(len(cls_ids)), cls_ids]

    keep = confs >= conf_thres
    if not np.any(keep):
        return [], [], []
    boxes_xywh = boxes_xywh[keep]
    confs = confs[keep]
    cls_ids = cls_ids[keep]

    x1 = np.clip((boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2 - pad_x) / ratio, 0, w)
    y1 = np.clip((boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2 - pad_y) / ratio, 0, h)
    bw = np.clip(boxes_xywh[:, 2] / ratio, 1, w - x1)
    bh = np.clip(boxes_xywh[:, 3] / ratio, 1, h - y1)
    boxes = np.stack([x1, y1, bw, bh], axis=1).tolist()
    return boxes, confs.tolist(), cls_ids.tolist()


def _merge_class_nms(all_boxes, all_confs, all_cids, orig_w: int, orig_h: int, conf_thres: float, iou_thres: float):
    detections = []
    if not all_boxes:
        return detections
    by_class = defaultdict(list)
    for b, c, cid in zip(all_boxes, all_confs, all_cids):
        by_class[int(cid)].append((b, float(c)))

    for cid, items in by_class.items():
        cls_boxes = [it[0] for it in items]
        cls_confs = [it[1] for it in items]
        indices = cv2.dnn.NMSBoxes(cls_boxes, cls_confs, conf_thres, iou_thres)
        if len(indices) == 0:
            continue
        for idx in np.array(indices).flatten():
            bx, by, bw, bh = cls_boxes[idx]
            detections.append({
                "clsId": cid,
                "label": CLASS_NAMES.get(cid, str(cid)),
                "conf": round(float(cls_confs[idx]), 4),
                "box": [
                    round(max(0.0, min(1.0, bx / orig_w)), 4),
                    round(max(0.0, min(1.0, by / orig_h)), 4),
                    round(max(0.001, min(1.0, bw / orig_w)), 4),
                    round(max(0.001, min(1.0, bh / orig_h)), 4),
                ],
            })
    detections.sort(key=lambda d: d["conf"], reverse=True)
    return detections


def run_onnx_detect(image_bytes: bytes, conf_thres: float = 0.35, iou_thres: float = 0.45, tile_mode: bool = False):
    if SESSION is None and MODEL_LOADING:
        MODEL_READY.wait(timeout=60)
    if SESSION is None:
        raise RuntimeError(MODEL_ERROR or "Model ONNX belum tersedia.")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Berkas gambar tidak dapat dibaca (gunakan JPG, PNG, atau WebP).")
    orig_h, orig_w = img.shape[:2]

    t0 = time.perf_counter()
    all_boxes, all_confs, all_cids = _infer_single_crop(img, conf_thres)

    if tile_mode and orig_w >= 320 and orig_h >= 320:
        tile_w = int(orig_w * 0.58)
        tile_h = int(orig_h * 0.58)
        xs = [0, max(0, orig_w - tile_w)]
        ys = [0, max(0, orig_h - tile_h)]
        for ty in ys:
            for tx in xs:
                crop = img[ty : ty + tile_h, tx : tx + tile_w]
                cb, cc, cids = _infer_single_crop(crop, conf_thres)
                for (bx, by, bw, bh), conf, cid in zip(cb, cc, cids):
                    all_boxes.append([bx + tx, by + ty, bw, bh])
                    all_confs.append(conf)
                    all_cids.append(cid)

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    detections = _merge_class_nms(all_boxes, all_confs, all_cids, orig_w, orig_h, conf_thres, iou_thres)
    return {
        "model": MODEL_DISPLAY_NAME,
        "modelFile": MODEL_FILE_NAME,
        "modelInputSize": MODEL_INPUT_SIZE,
        "tileMode": bool(tile_mode),
        "latencyMs": latency_ms,
        "imageSize": [orig_w, orig_h],
        "detections": detections,
    }


def _http_json(url: str, data: bytes | None = None, headers: dict | None = None, timeout: float = 10.0) -> dict:
    req_headers = {
        "User-Agent": f"SnapQC/{VERSION} (+https://snapqc.icxhnk.com)",
        "Accept": "application/json",
    }
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=req_headers, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_discovery(issuer: str) -> dict:
    now = time.time()
    cached = DISCOVERY_CACHE.get(issuer)
    if cached and cached["expires"] > now:
        return cached["doc"]
    doc = _http_json(f"{issuer}/.well-known/openid-configuration")
    if (doc.get("issuer") or "").rstrip("/") != issuer:
        raise ValueError("Issuer discovery tidak cocok dengan konfigurasi.")
    DISCOVERY_CACHE[issuer] = {"doc": doc, "expires": now + 300}
    return doc


def fetch_jwks(jwks_uri: str, force_refresh: bool = False) -> dict:
    now = time.time()
    cached = JWKS_CACHE.get(jwks_uri)
    if not force_refresh and cached and cached["expires"] > now:
        return cached["jwks"]
    jwks = _http_json(jwks_uri)
    JWKS_CACHE[jwks_uri] = {"jwks": jwks, "expires": now + 300}
    return jwks


def verify_access_token(access_token: str, cfg: dict, discovery: dict) -> dict:
    if jwt is None or PyJWK is None:
        raise RuntimeError("Dependensi PyJWT[crypto] belum terpasang untuk verifikasi RS256.")
    header = jwt.get_unverified_header(access_token)
    if header.get("alg") != "RS256":
        raise ValueError("Algoritma token harus RS256.")
    kid = header.get("kid")
    jwks_uri = discovery.get("jwks_uri")
    if not jwks_uri:
        raise ValueError("Discovery tidak menyertakan jwks_uri.")

    def _find_key(jwks_doc):
        for k in (jwks_doc.get("keys") or []):
            if not kid or k.get("kid") == kid:
                return PyJWK.from_dict(k).key
        return None

    jwks = fetch_jwks(jwks_uri)
    key = _find_key(jwks)
    if key is None:
        jwks = fetch_jwks(jwks_uri, force_refresh=True)
        key = _find_key(jwks)
    if key is None:
        raise ValueError("Kunci penandatangan JWKS tidak ditemukan.")

    claims = jwt.decode(
        access_token,
        key,
        algorithms=["RS256"],
        audience=cfg["client_id"],
        issuer=cfg["issuer"],
        options={"require": ["exp", "sub", "iss", "aud"]},
    )
    if not claims.get("sub"):
        raise ValueError("Claim sub wajib ada.")
    return claims


def has_required_role(claims: dict, required_roles: set[str]) -> bool:
    if not required_roles:
        return True
    roles = claims.get("roles")
    if isinstance(roles, str):
        role_set = {r.strip() for r in roles.split(",") if r.strip()}
    elif isinstance(roles, list):
        role_set = {str(r).strip() for r in roles if str(r).strip()}
    else:
        role_set = set()
    return bool(role_set.intersection(required_roles))


def make_pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def check_rate_limit(client_ip: str, limit_per_min: int) -> bool:
    now = time.time()
    cutoff = now - 60.0
    with STATE_LOCK:
        bucket = RATE_BUCKETS[client_ip]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= limit_per_min:
            return False
        bucket.append(now)
        return True


def prune_expired_state(cfg: dict):
    now = time.time()
    with STATE_LOCK:
        for st in [k for k, v in PENDING_OAUTH.items() if now - v["created_at"] > 600]:
            PENDING_OAUTH.pop(st, None)
    get_store(cfg).prune_sessions()


MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".json": "application/json; charset=utf-8",
}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _is_https(self, cfg: dict) -> bool:
        proto = (self.headers.get("X-Forwarded-Proto") or "").split(",")[0].strip().lower()
        return proto == "https" or cfg["app_origin"].startswith("https://")

    def _client_ip(self) -> str:
        fwd = (self.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
        return fwd or (self.client_address[0] if self.client_address else "unknown")

    def _get_cookie(self, name: str) -> str | None:
        raw = self.headers.get("Cookie")
        if not raw:
            return None
        jar = http.cookies.SimpleCookie()
        try:
            jar.load(raw)
        except Exception:
            return None
        morsel = jar.get(name)
        return morsel.value if morsel else None

    def _format_cookie(self, name: str, value: str, max_age: int, cfg: dict) -> str:
        parts = [f"{name}={value}", "Path=/", "HttpOnly", "SameSite=Lax", f"Max-Age={max_age}"]
        if self._is_https(cfg):
            parts.append("Secure")
        return "; ".join(parts)

    def _get_session(self, cfg: dict) -> tuple[str | None, dict | None]:
        sid = self._get_cookie("snapqc_session")
        if not sid:
            return None, None
        return sid, get_store(cfg).get_session(sid)

    def _require_access(self, cfg: dict) -> tuple[int, str, dict | None]:
        """Return (error_status, error_message, session_or_none).
        error_status == 0 means allowed.
        - hybrid (identity_enabled + public_demo_enabled): anyone allowed; signed-in session attached if present.
        - identity (identity_enabled only): valid signed-in session required; fail-closed if misconfigured.
        - public-demo: anyone allowed.
        - maintenance (both false): fail-closed 503.
        """
        _, sess = self._get_session(cfg)
        if cfg["identity_enabled"] and not cfg["public_demo_enabled"]:
            if not cfg["identity_configured"]:
                return 503, "Autentikasi SnapQC belum dikonfigurasi lengkap.", None
            if not sess:
                return 401, "Silakan masuk ke SnapQC terlebih dahulu.", None
            if not has_required_role({"roles": sess.get("roles", [])}, cfg["required_roles"]):
                return 403, "Akun ini tidak memiliki izin inspeksi di SnapQC.", None
            return 0, "", sess
        if cfg["public_demo_enabled"]:
            return 0, "", sess
        return 503, "SnapQC sedang dalam mode pemeliharaan.", None

    def _send_security_headers(self, cfg: dict):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        self.send_header("Permissions-Policy", "camera=(self), microphone=(), geolocation=()")
        if self._is_https(cfg):
            self.send_header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")

    def send_json(self, status: int, payload: dict, extra_headers: list[tuple[str, str]] | None = None):
        cfg = get_config()
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._send_security_headers(cfg)
        for k, v in (extra_headers or []):
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, status: int, data: bytes, content_type: str, extra_headers: list[tuple[str, str]] | None = None):
        cfg = get_config()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self._send_security_headers(cfg)
        for k, v in (extra_headers or []):
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def send_redirect(self, location: str, extra_headers: list[tuple[str, str]] | None = None):
        cfg = get_config()
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self._send_security_headers(cfg)
        for k, v in (extra_headers or []):
            self.send_header(k, v)
        self.end_headers()

    def _handle_oidc_start(self, cfg: dict):
        if not cfg["identity_enabled"] or not cfg["identity_configured"]:
            return self.send_json(503, {"error": "Layanan masuk akun SnapQC sedang tidak aktif."})
        prune_expired_state(cfg)
        try:
            discovery = fetch_discovery(cfg["issuer"])
            auth_endpoint = discovery.get("authorization_endpoint")
            if not auth_endpoint:
                raise ValueError("Discovery tidak menyertakan authorization_endpoint.")
        except Exception as exc:
            return self.send_json(503, {"error": f"Gagal menghubungi layanan masuk akun: {exc}"})

        state = secrets.token_urlsafe(24)
        verifier = secrets.token_urlsafe(48)
        challenge = make_pkce_challenge(verifier)
        with STATE_LOCK:
            PENDING_OAUTH[state] = {"code_verifier": verifier, "created_at": time.time()}

        params = urllib.parse.urlencode({
            "response_type": "code",
            "client_id": cfg["client_id"],
            "redirect_uri": cfg["redirect_uri"],
            "scope": "openid profile",
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        })
        cookie = self._format_cookie("snapqc_oauth_state", state, 600, cfg)
        return self.send_redirect(f"{auth_endpoint}?{params}", extra_headers=[("Set-Cookie", cookie)])

    def _handle_oidc_callback(self, cfg: dict, qs: dict):
        if not cfg["identity_enabled"] or not cfg["identity_configured"]:
            return self.send_json(503, {"error": "Layanan masuk akun SnapQC belum aktif."})
        if qs.get("error"):
            return self.send_json(403, {"error": f"Masuk ditolak: {qs['error'][0]}"})

        code = (qs.get("code") or [""])[0]
        state = (qs.get("state") or [""])[0]
        cookie_state = self._get_cookie("snapqc_oauth_state")
        if not code or not state or not cookie_state or not secrets.compare_digest(state, cookie_state):
            return self.send_json(400, {"error": "Parameter state atau code login tidak valid."})

        with STATE_LOCK:
            pending = PENDING_OAUTH.pop(state, None)
        if not pending or time.time() - pending["created_at"] > 600:
            return self.send_json(400, {"error": "Sesi login telah kedaluwarsa. Silakan ulangi masuk."})

        try:
            discovery = fetch_discovery(cfg["issuer"])
            token_endpoint = discovery.get("token_endpoint")
            if not token_endpoint:
                raise ValueError("Discovery tidak menyertakan token_endpoint.")
            form = urllib.parse.urlencode({
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": cfg["redirect_uri"],
                "client_id": cfg["client_id"],
                "client_secret": cfg["client_secret"],
                "code_verifier": pending["code_verifier"],
            }).encode("utf-8")
            token_res = _http_json(
                token_endpoint,
                data=form,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            access_token = token_res.get("access_token")
            if not access_token:
                raise ValueError("Token akses tidak dikembalikan.")
            claims = verify_access_token(access_token, cfg, discovery)
        except Exception as exc:
            return self.send_json(401, {"error": f"Verifikasi login gagal: {exc}"})

        if not has_required_role(claims, cfg["required_roles"]):
            clear_state = self._format_cookie("snapqc_oauth_state", "", 0, cfg)
            return self.send_json(
                403,
                {"error": "Akun ini belum memiliki akses ke SnapQC."},
                extra_headers=[("Set-Cookie", clear_state)],
            )

        email = claims.get("email") or ""
        display_name = claims.get("name") or claims.get("preferred_username") or (email.split("@")[0] if email else "Pengguna SnapQC")
        sid = secrets.token_urlsafe(32)
        get_store(cfg).put_session(sid, {
            "sub": claims["sub"],
            "name": display_name,
            "email": email,
            "roles": claims.get("roles") or [],
            "refresh_token": token_res.get("refresh_token"),
            "exp": float(claims.get("exp", time.time() + 3600)),
            "revoked": False,
        })
        clear_state = self._format_cookie("snapqc_oauth_state", "", 0, cfg)
        sess_cookie = self._format_cookie("snapqc_session", sid, 86400, cfg)
        return self.send_redirect("/", extra_headers=[("Set-Cookie", clear_state), ("Set-Cookie", sess_cookie)])

    def _handle_auth_refresh(self, cfg: dict):
        if not cfg["identity_enabled"] or not cfg["identity_configured"]:
            return self.send_json(503, {"error": "Layanan masuk akun SnapQC belum aktif."})
        sid, sess = self._get_session(cfg)
        if not sess or not sess.get("refresh_token"):
            return self.send_json(401, {"error": "Sesi tidak aktif atau tidak dapat diperbarui."})
        try:
            discovery = fetch_discovery(cfg["issuer"])
            form = urllib.parse.urlencode({
                "grant_type": "refresh_token",
                "refresh_token": sess["refresh_token"],
                "client_id": cfg["client_id"],
                "client_secret": cfg["client_secret"],
            }).encode("utf-8")
            token_res = _http_json(
                discovery["token_endpoint"],
                data=form,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            claims = verify_access_token(token_res["access_token"], cfg, discovery)
            if not has_required_role(claims, cfg["required_roles"]):
                get_store(cfg).revoke_session(sid)
                return self.send_json(403, {"error": "Akses SnapQC telah dicabut."})
            sess["exp"] = float(claims.get("exp", time.time() + 3600))
            sess["roles"] = claims.get("roles") or []
            if token_res.get("refresh_token"):
                sess["refresh_token"] = token_res["refresh_token"]
            get_store(cfg).put_session(sid, sess)
            return self.send_json(200, {"ok": True, "sub": claims["sub"], "roles": claims.get("roles") or []})
        except Exception as exc:
            return self.send_json(401, {"error": f"Pembaruan sesi gagal: {exc}"})

    def _handle_auth_logout(self, cfg: dict):
        sid = self._get_cookie("snapqc_session")
        refresh_token = get_store(cfg).revoke_session(sid) if sid else None
        if refresh_token and cfg["identity_enabled"] and cfg["identity_configured"]:
            try:
                discovery = fetch_discovery(cfg["issuer"])
                revoke_ep = discovery.get("revocation_endpoint")
                if revoke_ep:
                    form = urllib.parse.urlencode({
                        "token": refresh_token,
                        "client_id": cfg["client_id"],
                        "client_secret": cfg["client_secret"],
                    }).encode("utf-8")
                    _http_json(revoke_ep, data=form, headers={"Content-Type": "application/x-www-form-urlencoded"})
            except Exception:
                pass
        clear_sess = self._format_cookie("snapqc_session", "", 0, cfg)
        return self.send_json(200, {"ok": True}, extra_headers=[("Set-Cookie", clear_sess)])

    def do_GET(self):
        cfg = get_config()
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/healthz":
            db_stats = get_store(cfg).stats()
            return self.send_json(200, {
                "status": "ok",
                "modelLoaded": SESSION is not None,
                "modelLoading": MODEL_LOADING,
                "modelError": MODEL_ERROR,
                "modelName": MODEL_DISPLAY_NAME,
                "modelFile": MODEL_FILE_NAME,
                "modelInputSize": MODEL_INPUT_SIZE,
                "classes": CLASS_NAMES,
                "version": VERSION,
                "authMode": cfg["auth_mode"],
                "identityEnabled": cfg["identity_enabled"],
                "identityConfigured": cfg["identity_configured"],
                "publicDemoEnabled": cfg["public_demo_enabled"],
                "db": db_stats,
            })

        if path == "/api/auth/status":
            _, sess = self._get_session(cfg)
            user = (
                {"sub": sess["sub"], "name": sess["name"], "email": sess["email"], "roles": sess["roles"]}
                if sess
                else None
            )
            sso_ready = cfg["identity_enabled"] and cfg["identity_configured"]
            return self.send_json(200, {
                "authenticated": user is not None,
                "user": user,
                "authMode": cfg["auth_mode"],
                "identityEnabled": cfg["identity_enabled"],
                "identityConfigured": cfg["identity_configured"],
                "publicDemoEnabled": cfg["public_demo_enabled"],
                "loginUrl": "/api/auth/oidc/start" if sso_ready else None,
                "logoutUrl": "/api/auth/logout",
            })

        if path == "/api/auth/oidc/start":
            return self._handle_oidc_start(cfg)

        if path == "/login":
            if cfg["identity_enabled"] and cfg["identity_configured"]:
                return self._handle_oidc_start(cfg)
            return self.send_redirect("/")

        if path == "/api/auth/callback":
            return self._handle_oidc_callback(cfg, parse_qs(parsed.query))

        if path == "/api/metrics":
            v1 = json.loads(METRICS_V1_PATH.read_text(encoding="utf-8")) if METRICS_V1_PATH.exists() else None
            v2 = json.loads(METRICS_V2_PATH.read_text(encoding="utf-8")) if METRICS_V2_PATH.exists() else None
            return self.send_json(200, {
                "activeModel": MODEL_DISPLAY_NAME,
                "activeModelFile": MODEL_FILE_NAME,
                "activeInputSize": MODEL_INPUT_SIZE,
                "v1": v1,
                "v2": v2,
            })

        if path == "/api/feed":
            code, msg, _ = self._require_access(cfg)
            if code:
                return self.send_json(code, {"error": msg})
            store = get_store(cfg)
            return self.send_json(200, {"items": store.list_public_feed(limit=24)})

        if path == "/api/inspections":
            code, msg, sess = self._require_access(cfg)
            if code:
                return self.send_json(code, {"error": msg})
            owner_sub = sess["sub"] if sess else ""
            store = get_store(cfg)
            return self.send_json(200, {
                "scope": "personal" if owner_sub else "public-demo",
                "items": store.list_inspections(owner_sub=owner_sub),
                "stats": store.stats(owner_sub=owner_sub),
            })

        if path == "/api/inspections.csv":
            code, msg, sess = self._require_access(cfg)
            if code:
                return self.send_json(code, {"error": msg})
            owner_sub = sess["sub"] if sess else ""
            csv_data = get_store(cfg).export_csv(owner_sub=owner_sub).encode("utf-8-sig")
            return self.send_bytes(
                200,
                csv_data,
                "text/csv; charset=utf-8",
                extra_headers=[("Content-Disposition", 'attachment; filename="snapqc-inspections.csv"')],
            )

        if path == "/api/active-learning/export":
            code, msg, sess = self._require_access(cfg)
            if code:
                return self.send_json(code, {"error": msg})
            owner_sub = sess["sub"] if sess else ""
            al_data = json.dumps(get_store(cfg).export_active_learning(owner_sub=owner_sub), indent=2).encode("utf-8")
            return self.send_bytes(
                200,
                al_data,
                "application/json; charset=utf-8",
                extra_headers=[("Content-Disposition", 'attachment; filename="snapqc-active-learning.json"')],
            )

        if path.startswith("/api/inspections/"):
            code, msg, sess = self._require_access(cfg)
            if code:
                return self.send_json(code, {"error": msg})
            owner_sub = sess["sub"] if sess else ""
            insp_id = path.split("/", 3)[-1].strip()
            item = get_store(cfg).get_inspection(insp_id, owner_sub=owner_sub)
            if not item:
                return self.send_json(404, {"error": "Catatan inspeksi tidak ditemukan."})
            return self.send_json(200, item)

        if path in ("/", "/app", "/whats-new"):
            target = ROOT / "prototype" / "index.html"
        elif path in ("/portfolio", "/portfolio/"):
            target = ROOT / "portfolio" / "index.html"
        elif path == "/metrics.json":
            target = METRICS_V2_PATH if METRICS_V2_PATH.exists() else METRICS_V1_PATH
        else:
            rel = path.lstrip("/")
            cand1 = (ROOT / rel).resolve()
            cand2 = (ROOT / "prototype" / rel).resolve()
            target = cand1 if cand1.is_file() else cand2
            root_resolved = ROOT.resolve()
            if not target.is_relative_to(root_resolved) or not target.is_file():
                return self.send_json(404, {"error": "Berkas tidak ditemukan."})
            rel_parts = target.relative_to(root_resolved).parts
            if rel_parts[0] in {".venv", ".git", "datasets", "runs", "ml", "tests", "data"} or target.suffix.lower() in {".py", ".pt", ".env", ".sqlite", ".db"}:
                return self.send_json(404, {"error": "Berkas tidak ditemukan."})

        data = target.read_bytes()
        return self.send_bytes(200, data, MIME_TYPES.get(target.suffix.lower(), "application/octet-stream"))

    def do_POST(self):
        cfg = get_config()
        parsed = urlparse(self.path)
        if parsed.path == "/api/auth/refresh":
            return self._handle_auth_refresh(cfg)
        if parsed.path == "/api/auth/logout":
            return self._handle_auth_logout(cfg)

        if parsed.path == "/api/inspections":
            code, msg, sess = self._require_access(cfg)
            if code:
                return self.send_json(code, {"error": msg})
            if not check_rate_limit(self._client_ip() + ":save", cfg["rate_limit_per_min"]):
                return self.send_json(429, {"error": "Terlalu banyak permintaan simpan. Coba lagi sebentar."})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_JSON_BYTES:
                    return self.send_json(413, {"error": "Ukuran payload laporan melebihi batas (maks 3 MB)."})
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Format payload laporan harus objek JSON.")
                if not payload.get("modelVersion"):
                    payload["modelVersion"] = MODEL_DISPLAY_NAME
                owner_sub = sess["sub"] if sess else ""
                owner_name = sess["name"] if sess else ""
                saved = get_store(cfg).create_inspection(payload, owner_sub=owner_sub, owner_name=owner_name)
                return self.send_json(201, saved)
            except ValueError as exc:
                return self.send_json(400, {"error": str(exc)})
            except Exception as exc:
                return self.send_json(500, {"error": str(exc)})

        if parsed.path != "/api/detect":
            return self.send_json(404, {"error": "Endpoint tidak ditemukan."})

        code, msg, _ = self._require_access(cfg)
        if code:
            return self.send_json(code, {"error": msg})

        if not check_rate_limit(self._client_ip(), cfg["rate_limit_per_min"]):
            return self.send_json(429, {"error": "Terlalu banyak permintaan deteksi. Coba lagi dalam 1 menit."})

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_IMAGE_BYTES:
                return self.send_json(413, {"error": "Ukuran gambar harus antara 1 byte sampai 10 MB."})
            body = self.rfile.read(length)
            qs = parse_qs(parsed.query)
            conf = float(qs.get("conf", ["0.25"])[0])
            conf = max(0.05, min(0.95, conf))
            tile_mode = (qs.get("tile", ["0"])[0]).strip().lower() in {"1", "true", "yes"}
            result = run_onnx_detect(body, conf_thres=conf, tile_mode=tile_mode)
            return self.send_json(200, result)
        except ValueError as exc:
            return self.send_json(400, {"error": str(exc)})
        except Exception as exc:
            return self.send_json(500, {"error": str(exc)})

    def do_DELETE(self):
        cfg = get_config()
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/inspections/"):
            return self.send_json(404, {"error": "Endpoint tidak ditemukan."})
        code, msg, sess = self._require_access(cfg)
        if code:
            return self.send_json(code, {"error": msg})
        owner_sub = sess["sub"] if sess else ""
        insp_id = parsed.path.split("/", 3)[-1].strip()
        deleted = get_store(cfg).delete_inspection(insp_id, owner_sub=owner_sub)
        if not deleted:
            return self.send_json(404, {"error": "Catatan inspeksi tidak ditemukan."})
        return self.send_json(200, {"deleted": True, "id": insp_id})


def main():
    cfg = get_config()
    get_store(cfg)
    httpd = http.server.ThreadingHTTPServer((cfg["host"], cfg["port"]), Handler)
    print(
        f"SnapQC inference server listening: http://{cfg['host']}:{cfg['port']} "
        f"(version={VERSION}, authMode={cfg['auth_mode']}, db={cfg['db_path']}, loading ONNX model in background...)",
        flush=True,
    )
    threading.Thread(target=load_model, daemon=True).start()
    httpd.serve_forever()


if __name__ == "__main__":
    main()
