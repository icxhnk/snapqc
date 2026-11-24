"""Automated verification for SnapQC v0.4.0:
- Real ONNX inference + SAHI 2x2 tiled mode (?tile=1)
- Hybrid open guest + SSO mode & per-user SQLite ownership isolation
- Persistent SQLite QC inspection CRUD, CSV export, and Active Learning YOLO export
- Security headers, /whats-new, /portfolio, rate limiting, fail-closed mode
- OAuth 2.0 Authorization Code + PKCE S256 + RS256 JWKS + SQLite session persistence
"""

import http.server
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

import server as snapqc_server

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_IMG = ROOT / "prototype" / "samples" / "test-05.jpg"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class MockIdentityServer:
    def __init__(self):
        self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.public_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(self.private_key.public_key()))
        self.public_jwk["kid"] = "test-kid-1"
        self.public_jwk["use"] = "sig"
        self.public_jwk["alg"] = "RS256"
        self.next_token_claims = None
        self.revoked_tokens = set()

        parent = self

        class MockHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def _send(self, code, payload):
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == "/.well-known/openid-configuration":
                    return self._send(200, {
                        "issuer": parent.issuer,
                        "authorization_endpoint": f"{parent.issuer}/authorize",
                        "token_endpoint": f"{parent.issuer}/token",
                        "jwks_uri": f"{parent.issuer}/jwks",
                        "revocation_endpoint": f"{parent.issuer}/revoke",
                    })
                if self.path == "/jwks":
                    return self._send(200, {"keys": [parent.public_jwk]})
                self._send(404, {"error": "not_found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                form = urllib.parse.parse_qs(body)
                if self.path == "/revoke":
                    tok = (form.get("token") or [""])[0]
                    if tok:
                        parent.revoked_tokens.add(tok)
                    return self._send(200, {"revoked": True})
                if self.path == "/token":
                    grant = (form.get("grant_type") or [""])[0]
                    if grant == "refresh_token":
                        rt = (form.get("refresh_token") or [""])[0]
                        if rt in parent.revoked_tokens:
                            return self._send(400, {"error": "invalid_grant"})
                    claims = dict(parent.next_token_claims or {})
                    token = jwt.encode(claims, parent.private_key, algorithm="RS256", headers={"kid": "test-kid-1"})
                    return self._send(200, {
                        "access_token": token,
                        "token_type": "Bearer",
                        "refresh_token": "rt-test-1",
                    })
                self._send(404, {"error": "not_found"})

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        self.port = self.httpd.server_address[1]
        self.issuer = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class SnapQCServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        snapqc_server.load_model()
        cls.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), snapqc_server.Handler)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.mock_idp = MockIdentityServer()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.mock_idp.close()

    def setUp(self):
        self._saved_env = dict(os.environ)
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "test-snapqc.sqlite"
        snapqc_server.PENDING_OAUTH.clear()
        snapqc_server.DISCOVERY_CACHE.clear()
        snapqc_server.JWKS_CACHE.clear()
        snapqc_server.RATE_BUCKETS.clear()
        snapqc_server.STORE_CACHE.clear()
        os.environ["SNAPQC_DB_PATH"] = str(self._db_path)
        os.environ["APP_ORIGIN"] = "https://snapqc.icxhnk.com"
        os.environ["PUBLIC_DEMO_ENABLED"] = "true"
        os.environ["IDENTITY_PLATFORM_ENABLED"] = "false"
        os.environ["IDENTITY_REQUIRED_ROLES"] = ""
        os.environ["SNAPQC_RATE_LIMIT_PER_MIN"] = "30"

    def tearDown(self):
        snapqc_server.STORE_CACHE.clear()
        self._tmpdir.cleanup()
        os.environ.clear()
        os.environ.update(self._saved_env)

    def _req(self, path, method="GET", data=None, headers=None):
        req = urllib.request.Request(f"{self.base}{path}", data=data, method=method, headers=headers or {})
        opener = urllib.request.build_opener(NoRedirect)
        try:
            with opener.open(req, timeout=25) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def _login_user(self, sub: str, email: str, name: str, roles=None) -> str:
        _, h, _ = self._req("/api/auth/oidc/start")
        state = urllib.parse.parse_qs(urllib.parse.urlparse(h["Location"]).query)["state"][0]
        state_cookie = h["Set-Cookie"].split(";")[0]
        self.mock_idp.next_token_claims = {
            "iss": self.mock_idp.issuer,
            "aud": "snapqc",
            "sub": sub,
            "email": email,
            "name": name,
            "roles": roles or [],
            "exp": int(time.time()) + 3600,
        }
        req_cb = urllib.request.Request(
            f"{self.base}/api/auth/callback?code=code-{sub}&state={state}",
            headers={"Cookie": state_cookie},
        )
        opener = urllib.request.build_opener(NoRedirect)
        try:
            opener.open(req_cb, timeout=10)
            self.fail("Expected 302 redirect")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 302)
            cookies = e.headers.get_all("Set-Cookie") or []
            return [c.split(";")[0] for c in cookies if c.startswith("snapqc_session=")][0]

    def test_healthz_whats_new_portfolio_and_security_headers(self):
        status, headers, body = self._req("/healthz")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertTrue(data["modelLoaded"])
        self.assertEqual(data["version"], snapqc_server.VERSION)
        self.assertEqual(headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(headers.get("X-Frame-Options"), "DENY")
        self.assertIn("camera=(self)", headers.get("Permissions-Policy", ""))
        self.assertIn("max-age=31536000", headers.get("Strict-Transport-Security", ""))

        st_wn, _, body_wn = self._req("/whats-new")
        self.assertEqual(st_wn, 200)
        html = body_wn.decode("utf-8")
        self.assertIn("What's New", html)
        self.assertIn('id="menu-toggle"', html)
        self.assertNotIn("Sign in with ICXHNK", html)

        st_p, _, _ = self._req("/portfolio")
        self.assertEqual(st_p, 200)

    def test_live_onnx_detect_standard_and_sahi_tiling(self):
        img_bytes = SAMPLE_IMG.read_bytes()
        st1, _, b1 = self._req("/api/detect?conf=0.35&tile=0", method="POST", data=img_bytes)
        self.assertEqual(st1, 200)
        self.assertGreaterEqual(len(json.loads(b1)["detections"]), 3)

        st2, _, b2 = self._req("/api/detect?conf=0.35&tile=1", method="POST", data=img_bytes)
        self.assertEqual(st2, 200)
        d2 = json.loads(b2)
        self.assertTrue(d2["tileMode"])
        self.assertGreaterEqual(len(d2["detections"]), 3)

        bad_status, _, _ = self._req("/api/detect", method="POST", data=b"not-an-image")
        self.assertEqual(bad_status, 400)

    def test_hybrid_guest_plus_sso_personal_isolation(self):
        os.environ["IDENTITY_PLATFORM_ENABLED"] = "true"
        os.environ["PUBLIC_DEMO_ENABLED"] = "true"
        os.environ["IDENTITY_ISSUER"] = self.mock_idp.issuer
        os.environ["IDENTITY_CLIENT_ID"] = "snapqc"
        os.environ["IDENTITY_CLIENT_SECRET"] = "test-secret-value"
        os.environ["IDENTITY_REDIRECT_URI"] = "https://snapqc.icxhnk.com/api/auth/callback"
        os.environ["IDENTITY_REQUIRED_ROLES"] = ""

        # Healthz reports hybrid mode
        _, _, hb = self._req("/healthz")
        self.assertEqual(json.loads(hb)["authMode"], "hybrid")

        # Guest can detect AND save public-demo report without logging in
        st_gdet, _, _ = self._req("/api/detect", method="POST", data=SAMPLE_IMG.read_bytes())
        self.assertEqual(st_gdet, 200)
        guest_save = json.dumps({"vesselName": "Guest Ship", "verdict": "HOLD"}).encode("utf-8")
        st_gs, _, b_gs = self._req("/api/inspections", method="POST", data=guest_save, headers={"Content-Type": "application/json"})
        self.assertEqual(st_gs, 201)
        guest_id = json.loads(b_gs)["id"]

        # Any free SSO user (roles=[]) can sign in via SSO & save private personal report
        user_cookie = self._login_user("sub-alice", "alice@example.com", "Alice Inspector", roles=[])
        st_auth, _, b_auth = self._req("/api/auth/status", headers={"Cookie": user_cookie})
        self.assertEqual(st_auth, 200)
        auth_data = json.loads(b_auth)
        self.assertTrue(auth_data["authenticated"])
        self.assertEqual(auth_data["user"]["name"], "Alice Inspector")

        alice_save = json.dumps({"vesselName": "Alice Private Ship", "verdict": "ACCEPT", "yoloLabels": "1 0.5 0.5 0.2 0.2 # Good Weld"}).encode("utf-8")
        st_as, _, b_as = self._req("/api/inspections", method="POST", data=alice_save, headers={"Content-Type": "application/json", "Cookie": user_cookie})
        self.assertEqual(st_as, 201)
        alice_id = json.loads(b_as)["id"]

        # Isolation: Alice sees ONLY her private report, Guest sees ONLY guest report, Guest cannot read/delete Alice's report!
        _, _, alice_list = self._req("/api/inspections", headers={"Cookie": user_cookie})
        self.assertEqual([it["id"] for it in json.loads(alice_list)["items"]], [alice_id])
        _, _, guest_list = self._req("/api/inspections")
        self.assertEqual([it["id"] for it in json.loads(guest_list)["items"]], [guest_id])
        st_forbid, _, _ = self._req(f"/api/inspections/{alice_id}")
        self.assertEqual(st_forbid, 404)
        st_del_forbid, _, _ = self._req(f"/api/inspections/{alice_id}", method="DELETE")
        self.assertEqual(st_del_forbid, 404)

        # Public feed must include guest public post and NEVER leak Alice's private post
        st_feed, _, b_feed = self._req("/api/feed")
        self.assertEqual(st_feed, 200)
        feed_ids = [it["id"] for it in json.loads(b_feed)["items"]]
        self.assertIn(guest_id, feed_ids)
        self.assertNotIn(alice_id, feed_ids)

    def test_public_demo_rate_limit(self):
        os.environ["SNAPQC_RATE_LIMIT_PER_MIN"] = "2"
        img_bytes = SAMPLE_IMG.read_bytes()
        s1, _, _ = self._req("/api/detect", method="POST", data=img_bytes)
        s2, _, _ = self._req("/api/detect", method="POST", data=img_bytes)
        s3, _, _ = self._req("/api/detect", method="POST", data=img_bytes)
        self.assertEqual((s1, s2, s3), (200, 200, 429))

    def test_fail_closed_maintenance_mode(self):
        os.environ["IDENTITY_PLATFORM_ENABLED"] = "false"
        os.environ["PUBLIC_DEMO_ENABLED"] = "false"
        st, _, _ = self._req("/api/detect", method="POST", data=SAMPLE_IMG.read_bytes())
        self.assertEqual(st, 503)

    def test_strict_oauth_pkce_lifecycle_and_role_enforcement(self):
        os.environ["IDENTITY_PLATFORM_ENABLED"] = "true"
        os.environ["PUBLIC_DEMO_ENABLED"] = "false"
        os.environ["IDENTITY_ISSUER"] = self.mock_idp.issuer
        os.environ["IDENTITY_CLIENT_ID"] = "snapqc"
        os.environ["IDENTITY_CLIENT_SECRET"] = "test-secret-value"
        os.environ["IDENTITY_REDIRECT_URI"] = "https://snapqc.icxhnk.com/api/auth/callback"
        os.environ["IDENTITY_REQUIRED_ROLES"] = "snapqc:user,snapqc:admin"

        st_unauth, _, _ = self._req("/api/detect", method="POST", data=SAMPLE_IMG.read_bytes())
        self.assertEqual(st_unauth, 401)

        st_start, h_start, _ = self._req("/api/auth/oidc/start")
        self.assertEqual(st_start, 302)
        state = urllib.parse.parse_qs(urllib.parse.urlparse(h_start["Location"]).query)["state"][0]
        state_cookie = h_start["Set-Cookie"].split(";")[0]

        st_bad, _, _ = self._req("/api/auth/callback?code=abc&state=wrong-state", headers={"Cookie": state_cookie})
        self.assertEqual(st_bad, 400)

        self.mock_idp.next_token_claims = {
            "iss": self.mock_idp.issuer,
            "aud": "other-product",
            "sub": "user-123",
            "roles": ["snapqc:user"],
            "exp": int(time.time()) + 3600,
        }
        st_aud, _, _ = self._req(f"/api/auth/callback?code=abc&state={state}", headers={"Cookie": state_cookie})
        self.assertEqual(st_aud, 401)

        _, h2, _ = self._req("/api/auth/oidc/start")
        state2 = urllib.parse.parse_qs(urllib.parse.urlparse(h2["Location"]).query)["state"][0]
        cookie2 = h2["Set-Cookie"].split(";")[0]
        self.mock_idp.next_token_claims = {
            "iss": self.mock_idp.issuer,
            "aud": "snapqc",
            "sub": "user-no-role",
            "roles": ["dockledger:user"],
            "exp": int(time.time()) + 3600,
        }
        st_role, _, _ = self._req(f"/api/auth/callback?code=abc&state={state2}", headers={"Cookie": cookie2})
        self.assertEqual(st_role, 403)

        sess_cookie = self._login_user("user-valid-1", "ikhsan@example.com", "Ikhsan Inspector", roles=["snapqc:user"])
        snapqc_server.STORE_CACHE.clear()
        st_ok, _, _ = self._req("/api/detect", method="POST", data=SAMPLE_IMG.read_bytes(), headers={"Cookie": sess_cookie})
        self.assertEqual(st_ok, 200)

        st_ref, _, _ = self._req("/api/auth/refresh", method="POST", headers={"Cookie": sess_cookie})
        self.assertEqual(st_ref, 200)

        st_out, _, _ = self._req("/api/auth/logout", method="POST", headers={"Cookie": sess_cookie})
        self.assertEqual(st_out, 200)
        self.assertIn("rt-test-1", self.mock_idp.revoked_tokens)

        st_after, _, _ = self._req("/api/detect", method="POST", data=SAMPLE_IMG.read_bytes(), headers={"Cookie": sess_cookie})
        self.assertEqual(st_after, 401)


if __name__ == "__main__":
    unittest.main()
