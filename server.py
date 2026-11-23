"""SnapQC production & local inference server with real YOLO11n ONNX detection
and OAuth 2.0 Authorization Code + PKCE S256 support."""

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

try:
    import jwt
    from jwt import PyJWK
except ImportError:  # pragma: no cover - optional only in minimal local setups
    jwt = None
    PyJWK = None

ROOT = Path(__file__).resolve().parent
VERSION = "0.2.2"
METRICS_PATH = ROOT / "models" / "metrics-v1.json"
MAX_IMAGE_BYTES = 10 * 1024 * 1024

CLASS_NAMES = {0: "Bad Weld", 1: "Good Weld", 2: "Defect"}
SESSION = None
MODEL_LOADING = True
MODEL_ERROR = None
MODEL_READY = threading.Event()

# In-memory stores for OAuth PKCE state, server-side sessions, discovery/JWKS cache, and rate limits
PENDING_OAUTH = {}
SESSIONS = {}
DISCOVERY_CACHE = {}
JWKS_CACHE = {}
RATE_BUCKETS = defaultdict(deque)
STATE_LOCK = threading.Lock()


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_config() -> dict:
    issuer = (os.environ.get("IDENTITY_ISSUER") or "https://accounts.icxhnk.com").rstrip("/")
    client_id = (os.environ.get("IDENTITY_CLIENT_ID") or "snapqc").strip()
    client_secret = (os.environ.get("IDENTITY_CLIENT_SECRET") or "").strip()
    redirect_uri = (os.environ.get("IDENTITY_REDIRECT_URI") or "").strip()
    required_roles_raw = os.environ.get("IDENTITY_REQUIRED_ROLES", "snapqc:user,snapqc:admin")
    required_roles = {r.strip() for r in required_roles_raw.split(",") if r.strip()}
    identity_enabled = env_bool("IDENTITY_PLATFORM_ENABLED", False)
    public_demo_enabled = env_bool("PUBLIC_DEMO_ENABLED", True)
    app_origin = (os.environ.get("APP_ORIGIN") or "").rstrip("/")
    model_path = Path(os.environ.get("SNAPQC_MODEL_PATH") or (ROOT / "models" / "snapqc-welding-v1.onnx"))
    rate_limit = max(1, int(os.environ.get("SNAPQC_RATE_LIMIT_PER_MIN", "30")))
    identity_configured = bool(issuer and client_id and client_secret and redirect_uri)
    if identity_enabled:
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


def load_model():
    global SESSION, CLASS_NAMES, MODEL_LOADING, MODEL_ERROR
    t0 = time.perf_counter()
    cfg = get_config()
    model_path = cfg["model_path"]
    try:
        if METRICS_PATH.exists():
            meta = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            if isinstance(meta.get("classes"), dict):
                CLASS_NAMES = {int(k): v for k, v in meta["classes"].items()}
        if model_path.exists():
            import onnxruntime as ort

            SESSION = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            print(f"SnapQC ONNX model ready in {time.perf_counter() - t0:.1f}s ({model_path.name})", flush=True)
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


def run_onnx_detect(image_bytes: bytes, conf_thres: float = 0.35, iou_thres: float = 0.45):
    if SESSION is None and MODEL_LOADING:
        MODEL_READY.wait(timeout=60)
    if SESSION is None:
        raise RuntimeError(MODEL_ERROR or "Model ONNX belum tersedia. Jalankan ml/train_welding.py terlebih dahulu.")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Berkas gambar tidak dapat dibaca (gunakan JPG, PNG, atau WebP).")
    orig_h, orig_w = img.shape[:2]
    padded, ratio, pad_x, pad_y = letterbox(img, 640)
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob[None, ...])

    t0 = time.perf_counter()
    in_name = SESSION.get_inputs()[0].name
    out = SESSION.run(None, {in_name: blob})[0]
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    preds = out[0].T
    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]
    cls_ids = np.argmax(class_scores, axis=1)
    confs = class_scores[np.arange(len(cls_ids)), cls_ids]

    keep = confs >= conf_thres
    boxes_xywh = boxes_xywh[keep]
    confs = confs[keep]
    cls_ids = cls_ids[keep]

    detections = []
    if len(confs) > 0:
        x1 = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2 - pad_x) / ratio
        y1 = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2 - pad_y) / ratio
        bw = boxes_xywh[:, 2] / ratio
        bh = boxes_xywh[:, 3] / ratio
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        bw = np.clip(bw, 1, orig_w - x1)
        bh = np.clip(bh, 1, orig_h - y1)

        nms_boxes = np.stack([x1, y1, bw, bh], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(nms_boxes, confs.tolist(), conf_thres, iou_thres)
        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                bx, by, w, h = nms_boxes[idx]
                cid = int(cls_ids[idx])
                detections.append({
                    "clsId": cid,
                    "label": CLASS_NAMES.get(cid, str(cid)),
                    "conf": round(float(confs[idx]), 4),
                    "box": [
                        round(bx / orig_w, 4),
                        round(by / orig_h, 4),
                        round(w / orig_w, 4),
                        round(h / orig_h, 4),
                    ],
                })
    detections.sort(key=lambda d: d["conf"], reverse=True)
    return {
        "model": "SnapQC Welding v1 (YOLO11n ONNX)",
        "latencyMs": latency_ms,
        "imageSize": [orig_w, orig_h],
        "detections": detections,
    }


def _http_json(url: str, data: bytes | None = None, headers: dict | None = None, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data is not None else "GET")
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
        keys = jwks_doc.get("keys") or []
        for k in keys:
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


def prune_expired_state():
    now = time.time()
    with STATE_LOCK:
        for st in [k for k, v in PENDING_OAUTH.items() if now - v["created_at"] > 600]:
            PENDING_OAUTH.pop(st, None)
        for sid in [k for k, v in SESSIONS.items() if v.get("exp", 0) < now - 3600]:
            SESSIONS.pop(sid, None)


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

    def _get_session(self) -> tuple[str | None, dict | None]:
        sid = self._get_cookie("snapqc_session")
        if not sid:
            return None, None
        with STATE_LOCK:
            sess = SESSIONS.get(sid)
        if not sess or sess.get("revoked"):
            return sid, None
        if sess.get("exp", 0) <= time.time():
            return sid, None
        return sid, sess

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
            return self.send_json(503, {"error": "Autentikasi SnapQC sedang tidak aktif atau belum dikonfigurasi."})
        prune_expired_state()
        try:
            discovery = fetch_discovery(cfg["issuer"])
            auth_endpoint = discovery.get("authorization_endpoint")
            if not auth_endpoint:
                raise ValueError("Discovery tidak menyertakan authorization_endpoint.")
        except Exception as exc:
            return self.send_json(503, {"error": f"Gagal menghubungi layanan autentikasi: {exc}"})

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
            return self.send_json(503, {"error": "Autentikasi SnapQC belum aktif."})
        if qs.get("error"):
            return self.send_json(403, {"error": f"Masuk ditolak: {qs['error'][0]}"})

        code = (qs.get("code") or [""])[0]
        state = (qs.get("state") or [""])[0]
        cookie_state = self._get_cookie("snapqc_oauth_state")
        if not code or not state or not cookie_state or not secrets.compare_digest(state, cookie_state):
            return self.send_json(400, {"error": "Parameter state atau code OAuth tidak valid."})

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

        sid = secrets.token_urlsafe(32)
        with STATE_LOCK:
            SESSIONS[sid] = {
                "sub": claims["sub"],
                "name": claims.get("name") or claims.get("preferred_username") or "Inspector",
                "email": claims.get("email") or "",
                "roles": claims.get("roles") or [],
                "refresh_token": token_res.get("refresh_token"),
                "exp": float(claims.get("exp", time.time() + 3600)),
                "revoked": False,
            }
        clear_state = self._format_cookie("snapqc_oauth_state", "", 0, cfg)
        sess_cookie = self._format_cookie("snapqc_session", sid, 86400, cfg)
        return self.send_redirect("/", extra_headers=[("Set-Cookie", clear_state), ("Set-Cookie", sess_cookie)])

    def _handle_auth_refresh(self, cfg: dict):
        if not cfg["identity_enabled"] or not cfg["identity_configured"]:
            return self.send_json(503, {"error": "Autentikasi SnapQC belum aktif."})
        sid, sess = self._get_session()
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
                with STATE_LOCK:
                    SESSIONS.pop(sid, None)
                return self.send_json(403, {"error": "Akses SnapQC telah dicabut."})
            with STATE_LOCK:
                if sid in SESSIONS:
                    SESSIONS[sid]["exp"] = float(claims.get("exp", time.time() + 3600))
                    SESSIONS[sid]["roles"] = claims.get("roles") or []
                    if token_res.get("refresh_token"):
                        SESSIONS[sid]["refresh_token"] = token_res["refresh_token"]
            return self.send_json(200, {"ok": True, "sub": claims["sub"], "roles": claims.get("roles") or []})
        except Exception as exc:
            return self.send_json(401, {"error": f"Pembaruan sesi gagal: {exc}"})

    def _handle_auth_logout(self, cfg: dict):
        sid = self._get_cookie("snapqc_session")
        refresh_token = None
        if sid:
            with STATE_LOCK:
                sess = SESSIONS.pop(sid, None)
                if sess:
                    sess["revoked"] = True
                    refresh_token = sess.get("refresh_token")
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
            return self.send_json(200, {
                "status": "ok",
                "modelLoaded": SESSION is not None,
                "modelLoading": MODEL_LOADING,
                "modelError": MODEL_ERROR,
                "classes": CLASS_NAMES,
                "version": VERSION,
                "authMode": cfg["auth_mode"],
                "identityConfigured": cfg["identity_configured"],
            })

        if path == "/api/auth/status":
            _, sess = self._get_session()
            user = None
            if sess:
                user = {"sub": sess["sub"], "name": sess["name"], "roles": sess["roles"]}
            return self.send_json(200, {
                "authenticated": user is not None,
                "user": user,
                "authMode": cfg["auth_mode"],
                "identityEnabled": cfg["identity_enabled"],
                "identityConfigured": cfg["identity_configured"],
                "publicDemoEnabled": cfg["public_demo_enabled"],
                "loginUrl": "/api/auth/oidc/start" if cfg["identity_enabled"] else None,
            })

        if path == "/api/auth/oidc/start":
            return self._handle_oidc_start(cfg)

        if path == "/api/auth/callback":
            return self._handle_oidc_callback(cfg, parse_qs(parsed.query))

        if path in ("/", "/app", "/whats-new", "/login"):
            target = ROOT / "prototype" / "index.html"
        elif path in ("/portfolio", "/portfolio/"):
            target = ROOT / "portfolio" / "index.html"
        elif path == "/metrics.json":
            target = METRICS_PATH
        else:
            rel = path.lstrip("/")
            cand1 = (ROOT / rel).resolve()
            cand2 = (ROOT / "prototype" / rel).resolve()
            target = cand1 if cand1.is_file() else cand2
            root_resolved = ROOT.resolve()
            if not target.is_relative_to(root_resolved) or not target.is_file():
                return self.send_json(404, {"error": "Berkas tidak ditemukan."})
            rel_parts = target.relative_to(root_resolved).parts
            if rel_parts[0] in {".venv", ".git", "datasets", "runs", "ml", "tests"} or target.suffix.lower() in {".py", ".pt", ".env"}:
                return self.send_json(404, {"error": "Berkas tidak ditemukan."})

        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", MIME_TYPES.get(target.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self._send_security_headers(cfg)
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        cfg = get_config()
        parsed = urlparse(self.path)
        if parsed.path == "/api/auth/refresh":
            return self._handle_auth_refresh(cfg)
        if parsed.path == "/api/auth/logout":
            return self._handle_auth_logout(cfg)
        if parsed.path != "/api/detect":
            return self.send_json(404, {"error": "Endpoint tidak ditemukan."})

        # Access control: Identity gate OR public demo OR fail-closed maintenance
        if cfg["identity_enabled"]:
            if not cfg["identity_configured"]:
                return self.send_json(503, {"error": "Autentikasi SnapQC belum dikonfigurasi lengkap."})
            _, sess = self._get_session()
            if not sess:
                return self.send_json(401, {"error": "Silakan masuk ke SnapQC terlebih dahulu."})
            if not has_required_role({"roles": sess.get("roles", [])}, cfg["required_roles"]):
                return self.send_json(403, {"error": "Akun ini tidak memiliki izin inspeksi di SnapQC."})
        elif not cfg["public_demo_enabled"]:
            return self.send_json(503, {"error": "SnapQC sedang dalam mode pemeliharaan."})

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
            result = run_onnx_detect(body, conf_thres=conf)
            return self.send_json(200, result)
        except ValueError as exc:
            return self.send_json(400, {"error": str(exc)})
        except Exception as exc:
            return self.send_json(500, {"error": str(exc)})


def main():
    cfg = get_config()
    httpd = http.server.ThreadingHTTPServer((cfg["host"], cfg["port"]), Handler)
    print(
        f"SnapQC inference server listening: http://{cfg['host']}:{cfg['port']} "
        f"(version={VERSION}, authMode={cfg['auth_mode']}, loading ONNX model in background...)",
        flush=True,
    )
    threading.Thread(target=load_model, daemon=True).start()
    httpd.serve_forever()


if __name__ == "__main__":
    main()
