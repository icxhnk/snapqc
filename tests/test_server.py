"""Automated verification for SnapQC ONNX inference, security headers, What's New,
rate limiting, and OAuth 2.0 Authorization Code + PKCE S256 flows."""

import http.cookiejar
import http.server
import json
import os
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
    """Minimal local OAuth 2.0 + JWKS RS256 mock server for testing."""

    def __init__(self):
        self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.public_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(self.private_key.public_key()))
        self.public_jwk["kid"] = "test-kid-1"
        self.public_jwk["use"] = "sig"
        self.public_jwk["alg"] = "RS256"
        self.next_token_claims = None
        self.revoked_tokens = set()
        self.last_token_form = None

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
                        "authorization_endpoint": f"{parent.issuer}/oauth/authorize",
                        "token_endpoint": f"{parent.issuer}/oauth/token",
                        "jwks_uri": f"{parent.issuer}/oauth/jwks",
                        "revocation_endpoint": f"{parent.issuer}/oauth/revoke",
                    })
                if self.path == "/oauth/jwks":
                    return self._send(200, {"keys": [parent.public_jwk]})
                self._send(404, {"error": "not_found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                form = urllib.parse.parse_qs(body)
                if self.path == "/oauth/revoke":
                    tok = (form.get("token") or [""])[0]
                    if tok:
                        parent.revoked_tokens.add(tok)
                    return self._send(200, {"revoked": True})
                if self.path == "/oauth/token":
                    parent.last_token_form = form
                    grant = (form.get("grant_type") or [""])[0]
                    if grant == "refresh_token":
                        rt = (form.get("refresh_token") or [""])[0]
                        if rt in parent.revoked_tokens:
                            return self._send(400, {"error": "invalid_grant"})
                    claims = dict(parent.next_token_claims or {})
                    token = jwt.encode(
                        claims,
                        parent.private_key,
                        algorithm="RS256",
                        headers={"kid": "test-kid-1"},
                    )
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
        snapqc_server.PENDING_OAUTH.clear()
        snapqc_server.SESSIONS.clear()
        snapqc_server.DISCOVERY_CACHE.clear()
        snapqc_server.JWKS_CACHE.clear()
        snapqc_server.RATE_BUCKETS.clear()
        os.environ["APP_ORIGIN"] = "https://snapqc.icxhnk.com"
        os.environ["PUBLIC_DEMO_ENABLED"] = "true"
        os.environ["IDENTITY_PLATFORM_ENABLED"] = "false"
        os.environ["SNAPQC_RATE_LIMIT_PER_MIN"] = "30"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved_env)

    def _req(self, path, method="GET", data=None, headers=None):
        req = urllib.request.Request(f"{self.base}{path}", data=data, method=method, headers=headers or {})
        opener = urllib.request.build_opener(NoRedirect)
        try:
            with opener.open(req, timeout=15) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def test_healthz_whats_new_and_security_headers(self):
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

    def test_live_onnx_detect_and_validation(self):
        img_bytes = SAMPLE_IMG.read_bytes()
        status, _, body = self._req("/api/detect?conf=0.35", method="POST", data=img_bytes)
        self.assertEqual(status, 200)
        det = json.loads(body)
        self.assertGreaterEqual(len(det["detections"]), 3)
        labels = {d["label"] for d in det["detections"]}
        self.assertIn("Bad Weld", labels)
        self.assertIn("Good Weld", labels)

        bad_status, _, _ = self._req("/api/detect", method="POST", data=b"not-an-image")
        self.assertEqual(bad_status, 400)

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

    def test_oauth_pkce_full_lifecycle_and_rejections(self):
        os.environ["IDENTITY_PLATFORM_ENABLED"] = "true"
        os.environ["PUBLIC_DEMO_ENABLED"] = "false"
        os.environ["IDENTITY_ISSUER"] = self.mock_idp.issuer
        os.environ["IDENTITY_CLIENT_ID"] = "snapqc"
        os.environ["IDENTITY_CLIENT_SECRET"] = "test-secret-value"
        os.environ["IDENTITY_REDIRECT_URI"] = "https://snapqc.icxhnk.com/api/auth/callback"
        os.environ["IDENTITY_REQUIRED_ROLES"] = "snapqc:user,snapqc:admin"

        # Unauthenticated detect must be rejected with 401
        st_unauth, _, _ = self._req("/api/detect", method="POST", data=SAMPLE_IMG.read_bytes())
        self.assertEqual(st_unauth, 401)

        # Start OIDC flow -> 302 with PKCE S256 & state cookie
        st_start, h_start, _ = self._req("/api/auth/oidc/start")
        self.assertEqual(st_start, 302)
        loc = urllib.parse.urlparse(h_start["Location"])
        q = urllib.parse.parse_qs(loc.query)
        self.assertEqual(q["client_id"][0], "snapqc")
        self.assertEqual(q["code_challenge_method"][0], "S256")
        state = q["state"][0]
        state_cookie = h_start["Set-Cookie"].split(";")[0]

        # Bad state rejected with 400
        st_bad, _, _ = self._req(
            f"/api/auth/callback?code=abc&state=wrong-state",
            headers={"Cookie": state_cookie},
        )
        self.assertEqual(st_bad, 400)

        # Wrong audience rejected with 401
        self.mock_idp.next_token_claims = {
            "iss": self.mock_idp.issuer,
            "aud": "other-product",
            "sub": "user-123",
            "roles": ["snapqc:user"],
            "exp": int(time.time()) + 3600,
        }
        st_aud, _, _ = self._req(
            f"/api/auth/callback?code=abc&state={state}",
            headers={"Cookie": state_cookie},
        )
        self.assertEqual(st_aud, 401)

        # Start fresh flow & test missing entitlement role -> 403
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
        st_role, _, _ = self._req(
            f"/api/auth/callback?code=abc&state={state2}",
            headers={"Cookie": cookie2},
        )
        self.assertEqual(st_role, 403)

        # Start fresh flow & test valid login -> 302 + session cookie -> detect 200 -> refresh 200 -> logout -> 401
        _, h3, _ = self._req("/api/auth/oidc/start")
        state3 = urllib.parse.parse_qs(urllib.parse.urlparse(h3["Location"]).query)["state"][0]
        cookie3 = h3["Set-Cookie"].split(";")[0]
        self.mock_idp.next_token_claims = {
            "iss": self.mock_idp.issuer,
            "aud": "snapqc",
            "sub": "user-valid-1",
            "name": "Ikhsan Inspector",
            "roles": ["snapqc:user"],
            "exp": int(time.time()) + 3600,
        }
        req_cb = urllib.request.Request(
            f"{self.base}/api/auth/callback?code=valid-code&state={state3}",
            headers={"Cookie": cookie3},
        )
        opener = urllib.request.build_opener(NoRedirect)
        try:
            opener.open(req_cb, timeout=10)
            self.fail("Expected 302 redirect")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 302)
            cookies = e.headers.get_all("Set-Cookie") or []
            sess_cookie = [c.split(";")[0] for c in cookies if c.startswith("snapqc_session=")][0]

        # Authenticated detect works
        st_ok, _, _ = self._req(
            "/api/detect",
            method="POST",
            data=SAMPLE_IMG.read_bytes(),
            headers={"Cookie": sess_cookie},
        )
        self.assertEqual(st_ok, 200)

        # Refresh works
        st_ref, _, _ = self._req("/api/auth/refresh", method="POST", headers={"Cookie": sess_cookie})
        self.assertEqual(st_ref, 200)

        # Logout revokes session & refresh token
        st_out, _, _ = self._req("/api/auth/logout", method="POST", headers={"Cookie": sess_cookie})
        self.assertEqual(st_out, 200)
        self.assertIn("rt-test-1", self.mock_idp.revoked_tokens)

        # After logout, detect and refresh both fail with 401
        st_after, _, _ = self._req(
            "/api/detect",
            method="POST",
            data=SAMPLE_IMG.read_bytes(),
            headers={"Cookie": sess_cookie},
        )
        self.assertEqual(st_after, 401)
        st_ref_after, _, _ = self._req("/api/auth/refresh", method="POST", headers={"Cookie": sess_cookie})
        self.assertEqual(st_ref_after, 401)


if __name__ == "__main__":
    unittest.main()
