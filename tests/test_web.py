import unittest
import cv2
import numpy as np
from web_app import app,sessions

class WebTests(unittest.TestCase):
    def setUp(self):
        self.client=app.test_client()
        sessions.clear()
        self.jpeg=cv2.imencode('.jpg',np.zeros((120,160,3),np.uint8))[1].tobytes()
    def test_health_and_page(self):
        self.assertEqual(self.client.get('/healthz').json['status'],'ok')
        with self.client.get('/') as response:
            self.assertEqual(response.status_code,200)
    def test_image_validation(self):
        self.assertEqual(self.client.post('/api/frame',data=b'x',content_type='text/plain').status_code,415)
        self.assertEqual(self.client.post('/api/frame',data=b'x',content_type='image/jpeg').status_code,422)
        self.assertEqual(self.client.post('/api/frame',data=b'x'*750001,content_type='image/jpeg').status_code,413)
    def test_session_isolation_and_stop(self):
        a=self.client.post('/api/session').json['token'];b=self.client.post('/api/session').json['token']
        self.assertIsNot(sessions[a]['temporal'],sessions[b]['temporal'])
        response=self.client.post('/api/frame',data=self.jpeg,content_type='image/jpeg',headers={'X-Session-Token':a})
        self.assertEqual(response.status_code,200);self.assertEqual(response.json['faces'],[])
        self.client.delete('/api/session',headers={'X-Session-Token':a})
        self.assertNotIn(a,sessions);self.assertIn(b,sessions)
        self.assertEqual(self.client.post('/api/frame',data=self.jpeg,content_type='image/jpeg',headers={'X-Session-Token':a}).status_code,401)
    def test_bounded_sessions(self):
        for _ in range(32):self.assertEqual(self.client.post('/api/session').status_code,200)
        self.assertEqual(self.client.post('/api/session').status_code,503)

if __name__=='__main__':unittest.main()
