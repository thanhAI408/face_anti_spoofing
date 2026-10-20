"""Standalone anti-spoofing API; reuse the same inference pipeline as run.py."""
import argparse
import hashlib
import io
import os
from pathlib import Path
import secrets
import threading
import time

import cv2
import numpy as np
from flask import Flask, g, jsonify, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

from pipeline import BASE, Pipeline, TemporalDecision

MAX_BODY_BYTES = 2_000_000
MAX_IMAGE_EDGE = 1920


def create_app(model_path=None, labels_path=None, *, engine=None, api_key=None,
               threshold=0.8, session_ttl=30, max_sessions=128, clock=time.monotonic):
    """Load one model per process. Inject an engine/clock for contract tests."""
    if not 0.5 < threshold <= 1 or session_ttl <= 0 or max_sessions < 1:
        raise ValueError('Invalid threshold or session configuration')
    app = Flask(__name__, static_folder=None)
    app.config['MAX_CONTENT_LENGTH'] = MAX_BODY_BYTES
    model_path = Path(model_path or BASE / 'liveness_cam4k.h5')
    labels_path = Path(labels_path or BASE / 'le_cam4k.pickle')
    api_key = os.environ.get('ANTISPOOF_API_KEY', '') if api_key is None else api_key
    if engine is None:
        engine = Pipeline(model_path, labels_path, threshold=threshold)
        # Exclude the first inference allocation from normal request latency.
        engine.model(np.zeros((1, 32, 32, 3), dtype=np.float32), training=False)
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    else:
        digest = 'injected-test-engine'
    model_info = {'id': f'liveness-{digest[:12]}', 'sha256': digest,
                  'labels': list(engine.labels)}
    gate = threading.Lock()
    sessions = {}
    app.extensions['antispoof'] = {'engine': engine, 'gate': gate, 'sessions': sessions}

    def error(code, message, http_status):
        response = jsonify(error={'code': code, 'message': message}, request_id=g.request_id)
        response.status_code = http_status
        if http_status in (429, 503):
            response.headers['Retry-After'] = '1'
        return response

    @app.before_request
    def begin_request():
        g.started = time.perf_counter()
        g.request_id = secrets.token_hex(12)
        if request.path not in ('/v1/health', '/openapi.json') and api_key:
            actual = request.headers.get('Authorization', '')
            if not secrets.compare_digest(actual.encode('utf-8'), f'Bearer {api_key}'.encode('utf-8')):
                return error('unauthorized', 'A valid Bearer API key is required.', 401)

    @app.after_request
    def response_headers(response):
        response.headers['X-Request-ID'] = g.request_id
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return error('request_too_large' if exc.code == 413 else 'http_error',
                     exc.description, exc.code)

    @app.errorhandler(Exception)
    def unexpected_error(exc):
        app.logger.exception('Request %s failed', g.request_id)
        return error('internal_error', 'Inference failed; retry with a new session.', 500)

    @app.get('/v1/health')
    def health():
        return jsonify(status='ok', version='1.0', model=model_info, detector='yunet',
                       threshold=threshold, session_ttl_seconds=session_ttl,
                       max_image_edge=MAX_IMAGE_EDGE, max_body_bytes=MAX_BODY_BYTES,
                       authentication_required=bool(api_key))

    @app.get('/openapi.json')
    def openapi():
        return jsonify(build_openapi(require_auth=bool(api_key)))

    def prune(now):
        for token in list(sessions):
            if now - sessions[token]['seen'] >= session_ttl:
                del sessions[token]

    @app.post('/v1/sessions')
    def create_session():
        if not gate.acquire(blocking=False):
            return error('busy', 'Inference is busy. Retry with the newest frame.', 429)
        try:
            now = clock()
            prune(now)
            if len(sessions) >= max_sessions:
                return error('session_limit', 'Too many active sessions.', 503)
            token = secrets.token_urlsafe(24)
            sessions[token] = {'seen': now, 'temporal': TemporalDecision(threshold)}
            return jsonify(session_id=token, expires_in_seconds=session_ttl), 201
        finally:
            gate.release()

    @app.delete('/v1/sessions/<session_id>')
    def delete_session(session_id):
        if not gate.acquire(blocking=False):
            return error('busy', 'Inference is busy. Retry deletion.', 429)
        try:
            sessions.pop(session_id, None)
            return '', 204
        finally:
            gate.release()

    def analyze(session_id=None):
        # Do not build a queue of stale frames or decode many images concurrently.
        if not gate.acquire(blocking=False):
            return error('busy', 'Inference is busy. Retry with the newest frame.', 429)
        temporal = None
        completed = False
        try:
            now = clock()
            if session_id is not None:
                prune(now)
                session = sessions.get(session_id)
                if session is None:
                    return error('session_expired', 'Session is missing or expired.', 404)
                temporal = session['temporal']
            else:
                # Each image request has its own history and immediate model estimate.
                temporal = TemporalDecision(threshold, min_samples=1, min_seconds=0)
            if request.mimetype != 'image/jpeg':
                return error('unsupported_media_type', 'Send raw JPEG bytes as image/jpeg.', 415)
            data = request.get_data()
            try:
                with Image.open(io.BytesIO(data)) as im:
                    if im.format != 'JPEG' or min(im.size) < 32 or max(im.size) > MAX_IMAGE_EDGE:
                        return error('invalid_image', 'JPEG dimensions must be 32..1920 pixels.', 422)
                    im.verify()
                # Keep coordinates in the encoded pixel orientation (ignore EXIF rotation).
                pixels = cv2.imdecode(np.frombuffer(data, dtype=np.uint8),
                                      cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                if pixels is None:
                    raise ValueError('Image decode failed')
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                return error('invalid_image', 'Cannot decode the JPEG image.', 422)
            inference_start = time.perf_counter()
            faces = engine.process(pixels, now, temporal)
            processing_ms = (time.perf_counter() - inference_start) * 1000
            if session_id is not None:
                session['seen'] = clock()
            if not faces:
                status = 'no_face'
            elif len(faces) > 1:
                status = 'multiple_faces'
                # An identity transaction requires exactly one face in its image.
                temporal.tracks.clear()
            else:
                status = faces[0]['status']
            clean = [{k: v for k, v in face.items() if k not in ('landmarks', 'clipped')}
                     for face in faces]
            completed = True
            return jsonify(request_id=g.request_id,
                           mode='temporal' if session_id is not None else 'single_frame',
                           status=status, faces=clean,
                           image={'width': pixels.shape[1], 'height': pixels.shape[0],
                                  'sha256': hashlib.sha256(data).hexdigest()},
                           model=model_info, processing_ms=round(processing_ms, 3),
                           total_ms=round((time.perf_counter() - g.started) * 1000, 3))
        except Exception:
            if temporal is not None:
                temporal.tracks.clear()
            raise
        finally:
            if not completed and temporal is not None:
                temporal.tracks.clear()
            gate.release()

    @app.post('/v1/analyze')
    def analyze_image():
        return analyze()

    @app.post('/v1/sessions/<session_id>/frames')
    def analyze_frame(session_id):
        return analyze(session_id)

    return app


def build_openapi(require_auth=False):
    security = [{'bearerAuth': []}] if require_auth else []
    error = {'description': 'Structured error', 'content': {'application/json': {
        'schema': {'type': 'object', 'required': ['error', 'request_id'], 'properties': {
            'error': {'type': 'object', 'properties': {'code': {'type': 'string'},
                                                      'message': {'type': 'string'}}},
            'request_id': {'type': 'string'}}}}}}
    inference = {
        'summary': 'Evaluate a JPEG; this predicts spoof cues, not identity or authorization.',
        'security': security,
        'requestBody': {'required': True, 'content': {'image/jpeg': {
            'schema': {'type': 'string', 'format': 'binary'}}}},
        'responses': {'200': {'description': 'Model result; fail closed on undecided states.',
                              'content': {'application/json': {'schema': {'$ref': '#/components/schemas/Result'}}}},
                      **{str(code): error for code in (401, 404, 413, 415, 422, 429, 500)}}}
    statuses = ['real', 'fake', 'checking', 'uncertain', 'adjust', 'no_face', 'multiple_faces']
    return {'openapi': '3.0.3', 'info': {'title': 'Face Anti-Spoofing API', 'version': '1.0.0'},
            'servers': [{'url': 'http://127.0.0.1:8000'}],
            'paths': {
                '/v1/health': {'get': {'responses': {'200': {'description': 'Loaded model and limits'}}}},
                '/v1/analyze': {'post': inference},
                '/v1/sessions': {'post': {'security': security,
                    'responses': {'201': {'description': 'New session', 'content': {'application/json': {
                        'schema': {'type': 'object', 'properties': {'session_id': {'type': 'string'},
                            'expires_in_seconds': {'type': 'number'}}}}}}, '401': error, '429': error, '503': error}}},
                '/v1/sessions/{session_id}': {'parameters': [{'name': 'session_id', 'in': 'path',
                    'required': True, 'schema': {'type': 'string'}}], 'delete': {
                        'security': security, 'responses': {
                            '204': {'description': 'Session removed'}, '401': error, '429': error}}},
                '/v1/sessions/{session_id}/frames': {'parameters': [{'name': 'session_id', 'in': 'path',
                    'required': True, 'schema': {'type': 'string'}}], 'post': inference}},
            'components': {'securitySchemes': {'bearerAuth': {'type': 'http', 'scheme': 'bearer'}},
                'schemas': {'Result': {'type': 'object', 'properties': {
                    'request_id': {'type': 'string'}, 'mode': {'type': 'string', 'enum': ['single_frame', 'temporal']},
                    'status': {'type': 'string', 'enum': statuses},
                    'faces': {'type': 'array', 'items': {'type': 'object', 'properties': {
                        'box': {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 4, 'maxItems': 4},
                        'status': {'type': 'string'}, 'quality': {'type': 'string', 'nullable': True},
                        'scores': {'type': 'object', 'nullable': True, 'properties': {
                            'fake': {'type': 'number'}, 'real': {'type': 'number'}}},
                        'track_id': {'type': 'integer'}, 'detector_score': {'type': 'number'}}}},
                    'image': {'type': 'object', 'properties': {'width': {'type': 'integer'},
                        'height': {'type': 'integer'}, 'sha256': {'type': 'string'}}},
                    'model': {'type': 'object', 'properties': {'id': {'type': 'string'},
                        'sha256': {'type': 'string'}, 'labels': {'type': 'array', 'items': {'type': 'string'}}}},
                    'processing_ms': {'type': 'number'}, 'total_ms': {'type': 'number'}}}}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model', default=str(BASE / 'liveness_cam4k.h5'))
    parser.add_argument('--labels', default=str(BASE / 'le_cam4k.pickle'))
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--cpu-threads', type=int, default=2)
    args = parser.parse_args()
    if args.cpu_threads < 1 or not 0 < args.port < 65536:
        parser.error('Invalid port or CPU threads')
    if args.host not in ('127.0.0.1', 'localhost', '::1') and not os.environ.get('ANTISPOOF_API_KEY'):
        parser.error('Set ANTISPOOF_API_KEY before exposing the API on a network interface.')
    cv2.setNumThreads(args.cpu_threads)
    if not args.model.endswith('.tflite'):
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(args.cpu_threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    application = create_app(args.model, args.labels, threshold=args.threshold)
    from waitress import serve
    print(f'Anti-spoofing API ready: http://{args.host}:{args.port}/v1/health', flush=True)
    serve(application, host=args.host, port=args.port, threads=4,
          max_request_body_size=MAX_BODY_BYTES + 1, channel_timeout=30)


if __name__ == '__main__':
    main()
