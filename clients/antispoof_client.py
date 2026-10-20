"""Reusable client for the local anti-spoofing API.

The caller decides whether and when to run identity recognition. A ``real``
result from this service is not an authentication decision.
"""
import ipaddress
import os
import re
from urllib.parse import quote, urlsplit

import cv2
import numpy as np
import requests


class AntiSpoofAPIError(RuntimeError):
    def __init__(self, status_code, code, message, request_id=None):
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
        super().__init__(f"{code}: {message}")


def is_loopback_url(url):
    """Accept loopback destinations without consulting a proxy or DNS service."""
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return False
    if parsed.hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        return False


def encode_jpeg(frame, *, max_edge=960, quality=90):
    """Encode uint8 BGR camera data, resizing landscape AND portrait images."""
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError("frame must be a uint8 numpy BGR image")
    if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 32:
        raise ValueError("frame must have three channels and be at least 32x32")
    if not 32 <= max_edge <= 960 or not 1 <= quality <= 100:
        raise ValueError("max_edge must be 32..960 and quality must be 1..100")
    height, width = frame.shape[:2]
    scale = min(1.0, max_edge / max(height, width))
    if scale < 1.0:
        size = (round(width * scale), round(height * scale))
        if min(size) < 32:
            raise ValueError("image aspect ratio is too extreme for the API")
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("OpenCV could not encode the image")
    return encoded.tobytes()


class AntiSpoofClient:
    """One reusable connection for synchronous, sequential frame submissions.

    Localhost is the default. Set ``allow_remote=True`` deliberately to use an
    API hosted elsewhere. Requests never follow redirects or use proxy settings
    from the environment. The optional token is read from ANTISPOOF_API_KEY.
    Instances should not be shared across concurrent workers.
    """

    def __init__(self, url="http://127.0.0.1:8000", *, api_key=None,
                 timeout=(3.0, 15.0), allow_remote=False):
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("url must be an HTTP(S) address without credentials or query")
        if not allow_remote and not is_loopback_url(url):
            raise ValueError("remote API calls require explicit allow_remote=True")
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.http = requests.Session()
        self.http.trust_env = False
        self.http.headers.update({"Accept": "application/json"})
        token = os.environ.get("ANTISPOOF_API_KEY") if api_key is None else api_key
        if token:
            self.http.headers["Authorization"] = f"Bearer {token}"

    def _request(self, method, path, *, expected=200, **kwargs):
        response = self.http.request(method, self.url + path, timeout=self.timeout,
                                     allow_redirects=False, **kwargs)
        try:
            if response.status_code == expected == 204:
                return None
            try:
                result = response.json()
            except ValueError:
                if response.status_code >= 400:
                    code = 'request_too_large' if response.status_code == 413 else 'http_error'
                    raise AntiSpoofAPIError(response.status_code, code,
                                            'HTTP server rejected the request before JSON processing') from None
                raise AntiSpoofAPIError(response.status_code, "invalid_response",
                                        "API did not return JSON") from None
            if response.status_code != expected:
                error = result.get("error", {}) if isinstance(result, dict) else {}
                if not isinstance(error, dict):
                    error = {}
                request_id = result.get("request_id") if isinstance(result, dict) else None
                raise AntiSpoofAPIError(response.status_code, error.get("code", "http_error"),
                                        error.get("message", "API request failed"), request_id)
            if not isinstance(result, dict):
                raise AntiSpoofAPIError(response.status_code, "invalid_response",
                                        "API JSON must be an object")
            return result
        finally:
            response.close()

    @staticmethod
    def _jpeg(frame_or_jpeg):
        if isinstance(frame_or_jpeg, (bytes, bytearray, memoryview)):
            payload = bytes(frame_or_jpeg)
            if not payload:
                raise ValueError("JPEG payload cannot be empty")
            return payload
        return encode_jpeg(frame_or_jpeg)

    @staticmethod
    def _session_path(session_id):
        if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
            raise ValueError("session_id must be a URL-safe API session token")
        return "/v1/sessions/" + quote(session_id, safe="")

    def health(self):
        return self._request("GET", "/v1/health")

    def analyze(self, frame_or_jpeg):
        return self._request("POST", "/v1/analyze", data=self._jpeg(frame_or_jpeg),
                             headers={"Content-Type": "image/jpeg"})

    def create_session(self):
        result = self._request("POST", "/v1/sessions", expected=201, json={})
        session_id = result.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise AntiSpoofAPIError(201, "invalid_response", "API omitted session_id")
        return session_id

    def frame(self, session_id, frame_or_jpeg):
        return self._request("POST", self._session_path(session_id) + "/frames",
                             data=self._jpeg(frame_or_jpeg),
                             headers={"Content-Type": "image/jpeg"})

    def close_session(self, session_id):
        return self._request("DELETE", self._session_path(session_id), expected=204)

    def close(self):
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
