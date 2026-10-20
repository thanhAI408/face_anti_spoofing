"""Client contract tests with mocked HTTP; no camera, server or remote data."""
import os
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import requests

from clients.antispoof_client import AntiSpoofAPIError, AntiSpoofClient, encode_jpeg, is_loopback_url


class APIClientTests(unittest.TestCase):
    def setUp(self):
        self.http = Mock()
        self.http.headers = {}
        self.session_patch = patch("clients.antispoof_client.requests.Session", return_value=self.http)
        self.session_patch.start()
        self.addCleanup(self.session_patch.stop)
        self.env_patch = patch.dict(os.environ, {"ANTISPOOF_API_KEY": ""})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def response(self, status, payload=None):
        result = Mock(status_code=status)
        result.json.return_value = payload
        self.http.request.return_value = result
        return result

    def test_portrait_and_landscape_resize_preserve_aspect_and_bgr(self):
        for height, width in ((1600, 900), (900, 1600)):
            with self.subTest(height=height, width=width):
                frame = np.full((height, width, 3), (200, 80, 20), dtype=np.uint8)
                encoded = encode_jpeg(frame)
                decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(max(decoded.shape[:2]), 960)
                self.assertEqual(min(decoded.shape[:2]), 540)
                np.testing.assert_allclose(decoded[0, 0].astype(float), [200, 80, 20], atol=3)

    def test_invalid_frame_does_not_reach_http(self):
        client = AntiSpoofClient()
        for frame in (np.zeros((40, 40, 3)), np.zeros((40, 40), dtype=np.uint8),
                      np.zeros((20, 40, 3), dtype=np.uint8),
                      np.zeros((32, 2000, 3), dtype=np.uint8), b""):
            with self.subTest(shape=getattr(frame, "shape", None)):
                with self.assertRaises(ValueError):
                    client.analyze(frame)
        self.http.request.assert_not_called()

    def test_exact_jpeg_bytes_timeout_and_no_redirect(self):
        client = AntiSpoofClient(timeout=(1, 4))
        response = self.response(200, {"status": "no_face"})
        jpeg = encode_jpeg(np.zeros((40, 40, 3), dtype=np.uint8))
        self.assertEqual(client.analyze(memoryview(jpeg)), {"status": "no_face"})
        self.http.request.assert_called_once_with(
            "POST", "http://127.0.0.1:8000/v1/analyze", timeout=(1, 4),
            allow_redirects=False, data=jpeg, headers={"Content-Type": "image/jpeg"})
        self.assertFalse(self.http.trust_env)
        response.close.assert_called_once()

    def test_environment_key_and_explicit_override(self):
        with patch.dict(os.environ, {"ANTISPOOF_API_KEY": "test-only-token"}):
            AntiSpoofClient()
            self.assertEqual(self.http.headers["Authorization"], "Bearer test-only-token")
            self.http.headers = {}
            AntiSpoofClient(api_key="explicit-token")
            self.assertEqual(self.http.headers["Authorization"], "Bearer explicit-token")
            self.http.headers = {}
            AntiSpoofClient(api_key="")
            self.assertNotIn("Authorization", self.http.headers)

    def test_urls_require_intent_for_remote_and_reject_embedded_credentials(self):
        for url in ("http://localhost:8000", "http://127.0.0.1:8000", "http://[::1]:8000"):
            with self.subTest(url=url):
                self.assertTrue(is_loopback_url(url))
                AntiSpoofClient(url)
        for url in ("https://example.com", "http://localhost.example.com",
                    "http://127.0.0.1@external.example", "file:///tmp/example"):
            with self.subTest(url=url):
                self.assertFalse(is_loopback_url(url))
                with self.assertRaises(ValueError):
                    AntiSpoofClient(url)
        for url in ("http://user:password@localhost", "http://localhost?token=a", "http://localhost#fragment"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                AntiSpoofClient(url, allow_remote=True)
        AntiSpoofClient("https://example.com", allow_remote=True)
        self.http.request.assert_not_called()

    def test_session_lifecycle_and_reject_path_injection(self):
        client = AntiSpoofClient()
        self.response(201, {"session_id": "abc_123-XYZ"})
        session_id = client.create_session()
        self.assertEqual(session_id, "abc_123-XYZ")
        self.assertEqual(self.http.request.call_args.kwargs["json"], {})
        self.response(200, {"mode": "temporal", "status": "checking"})
        client.frame(session_id, b"jpeg-test-payload")
        self.assertEqual(self.http.request.call_args.args[:2],
                         ("POST", "http://127.0.0.1:8000/v1/sessions/abc_123-XYZ/frames"))
        response = self.response(204)
        self.assertIsNone(client.close_session(session_id))
        response.json.assert_not_called()
        self.http.request.reset_mock()
        for invalid in ("", "..", ".", "a/b", "a?b", "a#b", None):
            with self.subTest(session_id=invalid), self.assertRaises(ValueError):
                client.frame(invalid, b"image")
        self.http.request.assert_not_called()

    def test_structured_api_error_keeps_status_and_request_id(self):
        response = self.response(429, {"error": {"code": "busy", "message": "Try again"},
                                       "request_id": "req-1"})
        with self.assertRaises(AntiSpoofAPIError) as raised:
            AntiSpoofClient().analyze(b"image")
        self.assertEqual((raised.exception.status_code, raised.exception.code,
                          raised.exception.request_id), (429, "busy", "req-1"))
        response.close.assert_called_once()
        self.http.request.assert_called_once()  # No retries that accumulate old frames.

    def test_invalid_json_and_missing_session_fail_explicitly(self):
        client = AntiSpoofClient()
        response = self.response(200)
        response.json.side_effect = ValueError("HTML instead of JSON")
        with self.assertRaises(AntiSpoofAPIError) as raised:
            client.health()
        self.assertEqual(raised.exception.code, "invalid_response")
        response.close.assert_called_once()
        self.response(200, [])
        with self.assertRaises(AntiSpoofAPIError):
            client.health()
        self.response(201, {})
        with self.assertRaises(AntiSpoofAPIError):
            client.create_session()

    def test_health_context_cleanup_and_transport_timeout(self):
        response = self.response(200, {"status": "ok"})
        with AntiSpoofClient() as client:
            self.assertEqual(client.health()["status"], "ok")
        self.http.close.assert_called_once()
        response.close.assert_called_once()
        self.http.request.side_effect = requests.Timeout("timed out")
        with self.assertRaises(requests.Timeout):
            AntiSpoofClient().health()

    def test_server_rejection_without_json_keeps_http_error(self):
        response = self.response(413)
        response.json.side_effect = ValueError('server text response')
        with self.assertRaises(AntiSpoofAPIError) as raised:
            AntiSpoofClient().analyze(b'large-image')
        self.assertEqual(raised.exception.status_code, 413)
        self.assertEqual(raised.exception.code, 'request_too_large')
        response.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
