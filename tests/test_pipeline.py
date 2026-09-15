import unittest
import numpy as np
from pipeline import TemporalDecision,preprocess,FaceDetector,quality

class PipelineTests(unittest.TestCase):
    def test_solid_background_has_no_face(self):
        detector=FaceDetector()
        for color in [(220,170,40),(0,0,0),(255,255,255)]:
            self.assertEqual(detector.detect(np.full((480,640,3),color,dtype=np.uint8)),[])

    def test_resolution_and_bgr_preserved(self):
        out=preprocess(np.full((80,90,3),(255,128,0),dtype=np.uint8))
        self.assertEqual(out.shape,(32,32,3))
        np.testing.assert_allclose(out[0,0],[1,128/255,0])

    def test_decision_at_different_frame_rates(self):
        for fps in [10,30,60,120]:
            engine=TemporalDecision()
            t=engine.associate([(0,0,100,100)])[0]
            status=None
            for n in range(fps+1):
                status,_=engine.update(t,np.array([.01,.99]),n/fps)
            self.assertEqual(status,1)
            status,_=engine.update(t,np.array([.99,.01]),1+1/fps)
            self.assertEqual(status,'uncertain')

    def test_missing_and_bad_quality_reset(self):
        engine=TemporalDecision()
        t=engine.associate([(0,0,100,100)])[0]
        for n in range(10):engine.update(t,np.array([.01,.99]),n*.1)
        engine.update(t,None,1)
        self.assertEqual(engine.update(t,np.array([.01,.99]),1.1)[0],'checking')
        engine.associate([])
        replacement=engine.associate([(0,0,100,100)])[0]
        self.assertNotEqual(t.id,replacement.id)
        self.assertEqual(len(replacement.samples),0)

    def test_two_faces_do_not_share_history(self):
        engine=TemporalDecision()
        a,b=engine.associate([(0,0,100,100),(200,0,300,100)])
        engine.update(a,np.array([.99,.01]),0)
        engine.update(b,np.array([.01,.99]),0)
        b2,a2=engine.associate([(201,0,301,100),(1,0,101,100)])
        self.assertEqual((a.id,b.id),(a2.id,b2.id))
        self.assertEqual(int(a2.samples[0][1].argmax()),0)
        self.assertEqual(int(b2.samples[0][1].argmax()),1)

if __name__=='__main__': unittest.main()
