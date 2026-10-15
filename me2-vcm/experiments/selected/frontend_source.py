"""Deterministic whole-recording features with explicit capacity and mask."""
import numpy as np
from hey_swift.vcm.features import FeatureConfig,MelFeatures

CAPACITY_SAMPLES=96000
FRAMES=600
FLOOR=np.float32(np.log(1e-5))


class FullContextFeatures:
    def __init__(self):
        self.base=MelFeatures(FeatureConfig(window_seconds=6.))

    def transform(self,wave):
        wave=np.asarray(wave,np.float32)
        if wave.ndim!=1 or not len(wave) or not np.isfinite(wave).all():
            return None,None,{"status":"invalid_audio"}
        if len(wave)>CAPACITY_SAMPLES:
            return None,None,{"status":"over_capacity","input_samples":len(wave)}
        if float(np.ptp(wave))<=1e-8:
            return None,None,{"status":"flat_audio"}
        peak=float(np.abs(wave).max())
        if peak>1:
            wave=wave/peak
        n=1+(max(0,len(wave)-400)+159)//160
        padded=np.pad(wave,(0,400+(n-1)*160-len(wave)))
        frames=np.lib.stride_tricks.sliding_window_view(padded,400)[::160]
        assert len(frames)==n and n<=FRAMES
        power=np.abs(np.fft.rfft(frames*np.hanning(400),axis=1))**2
        mel=np.log(self.base._fb@power.T+1e-5).astype(np.float32)
        output=np.full((1,80,FRAMES),FLOOR,np.float32)
        mask=np.zeros((1,FRAMES),np.float32)
        output[:,:,:n]=mel;mask[:,:n]=1
        return output,mask,{"status":"ok","valid_frames":n,"input_samples":len(wave),
            "complete_waveform_retained":True,"complete_words_verified":False}
