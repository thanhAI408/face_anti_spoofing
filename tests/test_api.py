"""HTTP contracts tested without a trained model, camera, or network service."""
import hashlib
import io
import unittest

import cv2
import numpy as np
from PIL import Image

from api_app import MAX_BODY_BYTES, MAX_IMAGE_EDGE, create_app


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds=0.1):
        self.now += seconds


class FakeEngine:
    """Use real tracking/decision logic with predictable detector/model outputs."""
    labels = ['fake', 'real']

    def __init__(self):
        self.quality = None
        self.failure = None
        self.calls = 0

    def process(self, pixels, now, temporal):
        self.calls += 1
        if self.failure:
            raise self.failure
        # Uniform JPEG intensity defines the fixture: 0=none, 64=fake,
        # 128=real, 192=two faces, 224=ambiguous prediction.
        code = min((0, 64, 128, 192, 224), key=lambda value: abs(pixels.mean() - value))
        boxes = [] if code == 0 else [(20, 20, 100, 110)]
        if code == 192:
            boxes.append((130, 20, 210, 110))
        faces = []
        for box, track in zip(boxes, temporal.associate(boxes)):
            probs = np.array([0.99, 0.01] if code == 64 else [0.01, 0.99])
            if code == 224:
                probs = np.array([0.5, 0.5])
            status, mean = temporal.update(track, None if self.quality else probs, now)
            faces.append({
                'box': box, 'detector_score': 0.99, 'track_id': track.id,
                'status': self.labels[status] if isinstance(status, int) else status,
                'quality': self.quality,
                'scores': None if mean is None else {
                    label: float(mean[index]) for index, label in enumerate(self.labels)
                },
                'landmarks': np.zeros((5, 2)), 'clipped': False,
            })
        return faces


def jpeg(value=128, width=320, height=240):
    return cv2.imencode('.jpg', np.full((height, width, 3), value, np.uint8))[1].tobytes()


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.engine = FakeEngine()
        self.app = create_app(engine=self.engine, api_key='', clock=self.clock)
        self.client = self.app.test_client()
        self.real = jpeg(128)
        self.fake = jpeg(64)

    def post_image(self, data=None, session_id=None, client=None, **kwargs):
        path = '/v1/analyze' if session_id is None else f'/v1/sessions/{session_id}/frames'
        return (client or self.client).post(
            path, data=self.real if data is None else data, content_type='image/jpeg', **kwargs)

    def new_session(self, client=None):
        response = (client or self.client).post('/v1/sessions')
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json['expires_in_seconds'], 30)
        return response.json['session_id']

    def assert_error(self, response, status, code):
        self.assertEqual(response.status_code, status)
        self.assertEqual(response.json['error']['code'], code)
        self.assertEqual(response.json['request_id'], response.headers['X-Request-ID'])
        self.assertEqual(response.headers['Cache-Control'], 'no-store')

    def test_health_and_machine_readable_contract(self):
        response = self.client.get('/v1/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['status'], 'ok')
        self.assertEqual(response.json['model']['labels'], ['fake', 'real'])
        self.assertEqual(response.json['max_image_edge'], MAX_IMAGE_EDGE)
        self.assertFalse(response.json['authentication_required'])
        schema = self.client.get('/openapi.json').json
        self.assertEqual(schema['openapi'], '3.0.3')
        self.assertIn('/v1/sessions/{session_id}/frames', schema['paths'])
        statuses = schema['components']['schemas']['Result']['properties']['status']['enum']
        self.assertIn('multiple_faces', statuses)

    def test_single_frame_scores_metadata_and_exact_image_binding(self):
        response = self.post_image()
        self.assertEqual(response.status_code, 200)
        result = response.json
        self.assertEqual((result['mode'], result['status']), ('single_frame', 'real'))
        self.assertEqual(result['request_id'], response.headers['X-Request-ID'])
        self.assertEqual(result['image'], {
            'width': 320, 'height': 240, 'sha256': hashlib.sha256(self.real).hexdigest()})
        self.assertEqual(result['model']['labels'], ['fake', 'real'])
        self.assertEqual(set(result['faces'][0]['scores']), {'fake', 'real'})
        self.assertAlmostEqual(result['faces'][0]['scores']['real'], 0.99)
        self.assertNotIn('landmarks', result['faces'][0])
        self.assertNotIn('clipped', result['faces'][0])
        self.assertGreaterEqual(result['total_ms'], result['processing_ms'])
        self.assertEqual(response.headers['X-Content-Type-Options'], 'nosniff')
        second = self.post_image(self.fake)
        self.assertNotEqual(result['request_id'], second.json['request_id'])
        self.assertNotEqual(result['image']['sha256'], second.json['image']['sha256'])

    def test_single_frames_do_not_contaminate_sessions_or_each_other(self):
        token = self.new_session()
        for data, expected in ((self.real, 'real'), (self.fake, 'fake'), (self.real, 'real')):
            self.assertEqual(self.post_image(data).json['status'], expected)
        response = self.post_image(session_id=token)
        self.assertEqual(response.json['status'], 'checking')
        self.assertEqual(response.json['mode'], 'temporal')

    def test_temporal_real_and_fake_sessions_are_isolated(self):
        real_token, fake_token = self.new_session(), self.new_session()
        for index in range(7):
            real_result = self.post_image(self.real, real_token).json
            fake_result = self.post_image(self.fake, fake_token).json
            if index < 5:
                self.assertEqual(real_result['status'], 'checking')
                self.assertEqual(fake_result['status'], 'checking')
            self.clock.advance()
        self.assertEqual(real_result['status'], 'real')
        self.assertEqual(fake_result['status'], 'fake')
        self.assertEqual(self.post_image(self.fake, real_token).json['status'], 'uncertain')
        self.assertEqual(self.post_image(self.fake, fake_token).json['status'], 'fake')

    def test_quality_and_missing_face_clear_temporal_evidence(self):
        token = self.new_session()
        for _ in range(7):
            response = self.post_image(session_id=token)
            self.clock.advance()
        self.assertEqual(response.json['status'], 'real')
        self.engine.quality = 'Need more light'
        result = self.post_image(session_id=token).json
        self.assertEqual(result['status'], 'adjust')
        self.assertIsNone(result['faces'][0]['scores'])
        self.engine.quality = None
        self.assertEqual(self.post_image(session_id=token).json['status'], 'checking')
        self.assertEqual(self.post_image(jpeg(0), token).json['status'], 'no_face')
        self.assertEqual(self.app.extensions['antispoof']['sessions'][token]['temporal'].tracks, [])
        self.assertEqual(self.post_image(session_id=token).json['status'], 'checking')

    def test_multiple_faces_are_never_a_top_level_accept_and_clear_history(self):
        token = self.new_session()
        for _ in range(7):
            self.post_image(session_id=token)
            self.clock.advance()
        result = self.post_image(jpeg(192), token).json
        self.assertEqual(result['status'], 'multiple_faces')
        self.assertEqual(len(result['faces']), 2)
        self.assertEqual(self.app.extensions['antispoof']['sessions'][token]['temporal'].tracks, [])
        self.assertEqual(self.post_image(session_id=token).json['status'], 'checking')
        self.assertEqual(self.post_image(jpeg(192)).json['status'], 'multiple_faces')

    def test_portrait_720_by_1280_is_accepted(self):
        response = self.post_image(jpeg(width=720, height=1280))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['image']['width'], 720)
        self.assertEqual(response.json['image']['height'], 1280)

    def test_invalid_media_empty_corrupt_and_disguised_images(self):
        self.assert_error(self.client.post('/v1/analyze', json={}), 415, 'unsupported_media_type')
        self.assert_error(self.client.post('/v1/analyze', data=self.real,
                                          content_type='application/octet-stream'), 415,
                          'unsupported_media_type')
        for data in (b'', b'not a jpeg', b'\xff\xd8\xff'):
            self.assert_error(self.post_image(data), 422, 'invalid_image')
        png = io.BytesIO()
        Image.new('RGB', (100, 100)).save(png, format='PNG')
        self.assert_error(self.post_image(png.getvalue()), 422, 'invalid_image')
        self.assertEqual(self.engine.calls, 0)

    def test_body_and_image_dimensions_are_bounded(self):
        self.assert_error(self.post_image(b'x' * (MAX_BODY_BYTES + 1)), 413, 'request_too_large')
        for width, height in ((31, 100), (100, 31), (MAX_IMAGE_EDGE + 1, 100),
                              (100, MAX_IMAGE_EDGE + 1)):
            with self.subTest(width=width, height=height):
                self.assert_error(self.post_image(jpeg(width=width, height=height)), 422, 'invalid_image')
        self.assertEqual(self.engine.calls, 0)
        self.assertEqual(self.post_image(jpeg(width=32, height=32)).status_code, 200)

    def test_session_expiration_renewal_and_deletion(self):
        token = self.new_session()
        self.clock.advance(29)
        self.assertEqual(self.post_image(session_id=token).status_code, 200)
        self.clock.advance(29)
        self.assertEqual(self.post_image(session_id=token).status_code, 200)
        self.clock.advance(30)
        self.assert_error(self.post_image(session_id=token), 404, 'session_expired')
        token = self.new_session()
        self.assertEqual(self.client.delete(f'/v1/sessions/{token}').status_code, 204)
        self.assertEqual(self.client.delete(f'/v1/sessions/{token}').status_code, 204)
        self.assert_error(self.post_image(session_id=token), 404, 'session_expired')

    def test_session_count_is_bounded_and_expired_capacity_is_reused(self):
        app = create_app(engine=self.engine, api_key='', max_sessions=2, clock=self.clock)
        client = app.test_client()
        self.new_session(client)
        self.new_session(client)
        response = client.post('/v1/sessions')
        self.assert_error(response, 503, 'session_limit')
        self.assertEqual(response.headers['Retry-After'], '1')
        self.clock.advance(30)
        self.new_session(client)
        self.assertEqual(len(app.extensions['antispoof']['sessions']), 1)

    def test_busy_inference_rejects_queue_and_session_mutation(self):
        token = self.new_session()
        gate = self.app.extensions['antispoof']['gate']
        with gate:
            responses = [self.post_image(), self.post_image(session_id=token),
                         self.client.post('/v1/sessions'), self.client.delete(f'/v1/sessions/{token}')]
            for response in responses:
                self.assert_error(response, 429, 'busy')
                self.assertEqual(response.headers['Retry-After'], '1')
            self.assertEqual(self.client.get('/v1/health').status_code, 200)
        self.assertEqual(self.engine.calls, 0)
        self.assertEqual(self.post_image(session_id=token).status_code, 200)

    def test_optional_bearer_authentication(self):
        client = create_app(engine=self.engine, api_key='contract-test-key', clock=self.clock).test_client()
        self.assertTrue(client.get('/v1/health').json['authentication_required'])
        self.assertEqual(client.get('/openapi.json').status_code, 200)
        for headers in ({}, {'Authorization': 'Bearer incorrect'}, {'Authorization': 'contract-test-key'},
                        {'Authorization': 'Bearer café'}):
            self.assert_error(self.post_image(client=client, headers=headers), 401, 'unauthorized')
            self.assert_error(client.post('/v1/sessions', headers=headers), 401, 'unauthorized')
        response = self.post_image(client=client, headers={'Authorization': 'Bearer contract-test-key'})
        self.assertEqual(response.status_code, 200)
        schema = client.get('/openapi.json').json
        self.assertEqual(schema['paths']['/v1/analyze']['post']['security'], [{'bearerAuth': []}])

    def test_bad_frame_resets_existing_temporal_history(self):
        token = self.new_session()
        for bad in (b'bad jpeg', jpeg(width=2000, height=100)):
            for _ in range(7):
                self.post_image(session_id=token)
                self.clock.advance()
            self.assertEqual(self.post_image(session_id=token).json['status'], 'real')
            self.assertEqual(self.post_image(bad, token).status_code, 422)
            self.assertEqual(self.post_image(session_id=token).json['status'], 'checking')

    def test_ambiguous_single_frame_is_not_accepted(self):
        self.assertEqual(self.post_image(jpeg(224)).json['status'], 'uncertain')

    def test_inference_failure_clears_history_releases_lock_and_hides_details(self):
        token = self.new_session()
        self.post_image(session_id=token)
        self.engine.failure = RuntimeError('private model path must not appear in response')
        with self.assertLogs(self.app.logger, level='ERROR'):
            response = self.post_image(session_id=token)
        self.assert_error(response, 500, 'internal_error')
        self.assertNotIn('private model path', response.get_data(as_text=True))
        self.assertEqual(self.app.extensions['antispoof']['sessions'][token]['temporal'].tracks, [])
        self.engine.failure = None
        self.assertEqual(self.post_image(session_id=token).json['status'], 'checking')


if __name__ == '__main__':
    unittest.main()
