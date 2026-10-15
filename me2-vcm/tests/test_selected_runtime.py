"""Host tests for frozen preprocessing, rejection and the real command loop."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from contextlib import contextmanager

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def model(profile="selected"):
    from hey_swift.vcm.inference import VCMInference
    p=json.loads((ROOT/"configs/assistant_profiles.json").read_text())["profiles"][profile]
    return VCMInference(str(ROOT/p["config"]),str(ROOT/p["model"]),str(ROOT/p["labels"]))


def wave(n=16000):
    return (.1*np.sin(np.arange(n)*2*np.pi*440/16000)).astype(np.float32)


def test_six_second_frontend_boundaries_and_legacy_fallback():
    selected=model()
    x,mask,receipt=selected.feats.transform(wave(96000))
    assert x.shape==(1,80,600) and mask.sum()==599
    assert receipt["complete_waveform_retained"]
    assert selected.predict(wave(96001))["rejection_reason"]=="over_capacity"
    assert selected.predict(np.zeros(16000,np.float32))["accepted"] is False
    assert selected.predict(np.array([np.nan],np.float32))["accepted"] is False
    fallback=model("run043")
    assert fallback.tau==.875 and fallback.frontend=="legacy_center_crop"
    assert fallback.feats.transform(wave()).shape==(1,80,198)


def test_strict_command_preparation_never_crops():
    from hey_swift.audio.endpoint import prepare_command_audio
    audio=wave(96001)
    meta=dict(reason="end_silence",speech_start_sample=2000,speech_end_sample=80000)
    actual,reason=prepare_command_audio(audio,meta,6,preserve_full=True)
    assert actual is None and reason=="command_too_long"
    for metadata,expected in ((dict(reason="max_duration"),"command_incomplete"),
                              (dict(reason="end_silence",overflow=True),"capture_overflow")):
        assert prepare_command_audio(wave(),metadata,6,preserve_full=True)[1]==expected


def test_capture_preserves_training_length_pause_and_ends_dynamically(monkeypatch):
    from hey_swift.integration import e2e_loop as loop
    import hey_swift.audio.capture as capture
    clock=[0.]
    # Calibration, start delay, speech, 1.08s internal pause, speech, endpoint.
    levels=[.001]*4+[.001]*5+[.1]*10+[.001]*27+[.1]*10+[.001]*40
    class Stream:
        def __init__(self,**kwargs): self.i=0
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def read(self,n):
            value=levels[self.i] if self.i<len(levels) else .001
            self.i+=1;clock[0]+=.04
            return np.full(n,value,np.float32).tobytes(),False
    class Resampler:
        def __init__(self,*args): pass
        def push(self,x): return x
    monkeypatch.setitem(sys.modules,"sounddevice",SimpleNamespace(RawInputStream=Stream))
    monkeypatch.setattr(capture,"ContinuousResampler",Resampler)
    monkeypatch.setattr(loop.time,"monotonic",lambda:clock[0])
    args=SimpleNamespace(vad_frame_ms=40,command_seconds=7.4,min_command_seconds=.5,
        speech_start_ms=120,pre_roll_ms=200,end_silence_ms=1200,
        speech_start_timeout=3.,vad_noise_seconds=.16,vad_margin_db=10,vad_min_speech_db=-45)
    audio,meta=loop._capture_command_vad(0,args,rate=16000)
    assert meta["reason"]=="end_silence"
    assert np.count_nonzero(audio>.05)==20*640
    assert meta["capture_seconds"]<6
    assert 1.19<=meta["endpoint_wait_seconds"]<=1.25


@pytest.mark.parametrize("intent,slot,rejected",[("TEMPERATURE","22",False),("TEMPERATURE","REJECT",True),("OUT_OF_SCOPE",None,True)])
def test_real_wake_capture_model_router_idle_loop(monkeypatch,intent,slot,rejected):
    from hey_swift.integration import e2e_loop as loop
    from hey_swift.integration.router import ActionRouter,INTENTS
    from hey_swift.actions.services.information import ActionResult
    import hey_swift.integration.runtime as runtime
    import hey_swift.audio.capture as capture
    import hey_swift.audio.recovery as recovery
    import hey_swift.vcm.inference as inference
    import hey_swift.wakeword.inference as wake
    events=[]
    def emit_print(text,*args,**kwargs):
        try: events.append(json.loads(text))
        except (ValueError,TypeError): pass
    monkeypatch.setattr(loop,"print",emit_print,raising=False)
    class Service:
        def execute(self,name,value=None,**kwargs):
            assert any(x.get("event")=="benchmark_decision" for x in events)
            events.append(dict(event="mock_service",intent=name,value=value))
            return ActionResult(name,"ok","mock")
    monkeypatch.setattr(runtime,"build_router",lambda **kwargs:ActionRouter({g:Service() for g in set(INTENTS.values())}))
    monkeypatch.setattr(runtime,"prepare_workspace",lambda:None)
    monkeypatch.setitem(sys.modules,"sounddevice",SimpleNamespace(PortAudioError=RuntimeError))
    monkeypatch.setattr(capture,"capture_supported_rate",lambda device:(16000,1600))
    mic=SimpleNamespace(resolve=lambda:0,name="mock USB",hostapi="mock",healthy=lambda:None)
    monkeypatch.setattr(recovery,"StableMicrophone",lambda *args,**kwargs:mic)
    monkeypatch.setattr(wake,"WakeWordDetector",lambda *args,**kwargs:SimpleNamespace(reset=lambda:None,gate=SimpleNamespace(threshold=.3)))
    actual=model()
    arrays={o.name:np.zeros((1,int(o.shape[-1])),np.float32) for o in actual.sess.get_outputs()}
    arrays["intent_logits"][0,actual.intent_labels.index(intent)]=12
    if slot is not None:
        arrays["slot_"+intent][0,actual.slot_sets[intent].index(slot)]=12
    actual.sess=SimpleNamespace(run=lambda *args,**kwargs:list(arrays.values()))
    monkeypatch.setattr(inference,"VCMInference",lambda *args,**kwargs:actual)
    wakes=[0]
    def listen(*args,**kwargs):
        wakes[0]+=1
        if wakes[0]>1: raise KeyboardInterrupt()
        events.append(dict(event="wake"))
        return 0.
    monkeypatch.setattr(loop,"_listen_for_wake",listen)
    def command(*args,before_ready=None,**kwargs):
        if before_ready: before_ready()
        return wave(),dict(reason="end_silence",speech_started=True,speech_start_sample=0,
            speech_end_sample=16000,_speech_end_monotonic=0.,overflow=False)
    monkeypatch.setattr(loop,"_capture_command_vad",command)
    @contextmanager
    def duck(*args,**kwargs): yield lambda:None
    monkeypatch.setattr(loop,"_duck_music",duck)
    monkeypatch.setattr(loop,"_play_ack",lambda *args,**kwargs:None)
    status=loop.main(["--vcm-onnx","unused","--vcm-config",str(ROOT/"configs/vcm_selected.yaml"),
        "--vcm-labels",str(ROOT/"models/vcm/selected.labels.json"),"--no-ack","--playback-tail-seconds","0",
        "--benchmark-log","--log-format","json"])
    assert status==0 and wakes[0]==2
    records=[e for e in events if e.get("event")=="benchmark_decision"]
    assert len(records)==1 and records[0]["accepted"] is not rejected
    assert records[0]["audio_ms"]==1000.
    services=[e for e in events if e.get("event")=="mock_service"]
    assert len(services)==(0 if rejected else 1)
    assert sum(e.get("status")=="listening" for e in events)==2
