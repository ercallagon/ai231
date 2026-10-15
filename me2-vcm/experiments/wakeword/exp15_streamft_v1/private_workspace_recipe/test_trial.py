import sys
from pathlib import Path
import unittest
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import trial as T
from hey_swift.wakeword.inference import WakeWordDetector,GateConfig


class TrialTests(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(T.label_window(0,16000,2000,15000),0)
        self.assertEqual(T.label_window(0,16000,12000,18000),-100)
        self.assertEqual(T.label_window(16000,32000,12000,18000),-100)
        self.assertEqual(T.label_window(20000,36000,12000,18000),1)
        self.assertEqual(T.label_window(0,16000),1)

    def test_fresh_scores_and_runtime_equivalence(self):
        meta=dict(samples=np.arange(16000,160000+1,4000).tolist(),rms=[.1]*37)
        p=np.random.default_rng(3).uniform(0,1,37)
        p[2:5]=[0,.99,0]
        expected=T.events(p,meta,.5)
        d=object.__new__(WakeWordDetector);d.win=16000;d.hop_samples=d.subhop_samples=4000
        d.n_subwin=1;d.sustain=2;d.gate=GateConfig(threshold=.5,multiwin=False)
        mapping=dict(zip(meta["samples"],p))
        d._score=lambda:(mapping[d._state.samples],1-mapping[d._state.samples],True)
        for chunk in (1600,1733,160000):
            d.reset();actual=[]
            for i in range(0,160000,chunk):actual.extend(d.process(np.zeros(min(chunk,160000-i),np.float32))[0])
            self.assertEqual([round(e["t"]*16000) for e in actual],expected)

    def test_retained_score_is_not_fresh_evidence(self):
        p=np.array([0.,0.,.99,0.,0.,0.])
        m=dict(samples=np.arange(16000,16000+6*4000,4000).tolist(),rms=[.1]*6)
        self.assertEqual(T.events(p,m,.5),[])
        maximum=np.max(np.stack([np.r_[np.zeros(k),p][:len(p)] for k in range(4)]),axis=0)
        self.assertGreater(len(T.events(maximum,m,.5)),0)

    def test_silence(self):
        m=dict(samples=[16000,20000],rms=[0.,0.])
        self.assertEqual(T.events(np.ones(2),m,.5),[])


if __name__=="__main__":unittest.main()
