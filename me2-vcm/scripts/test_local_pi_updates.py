"""Offline route and synthetic capture regression checks; no external effects."""
import json
import sys
import unittest
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hey_swift.integration import e2e_loop as loop
from hey_swift.integration.router import ActionRouter, INTENTS, SLOTS, Prediction
from hey_swift.actions.services.information import ActionResult

class Tests(unittest.TestCase):
    def test_routes(self):
        label_files = sorted((ROOT / "models/vcm").glob("*.labels.json"))
        self.assertTrue(label_files, "No packaged runtime labels found")
        label_sets = []
        for path in label_files:
            labels = json.loads(path.read_text())
            slots = labels.get("slot_sets", labels.get("slots"))
            self.assertIsInstance(slots, dict, f"Missing slot labels: {path}")
            label_sets.append({"intents": labels["intents"], "slot_sets": slots})
        calls = []
        class Service:
            def __init__(self, group): self.group = group
            def execute(self, intent, slot=None, **kwargs):
                calls.append((self.group, intent, slot))
                return ActionResult(intent, "ok", "checked")
        with ActionRouter({group: Service(group) for group in set(INTENTS.values())}) as router:
            routes = [(intent, value) for labels in label_sets for intent in labels["intents"]
                      for value in labels["slot_sets"].get(intent, [None])]
            for intent, value in routes:
                if intent == "OUT_OF_SCOPE": continue
                self.assertIn(intent, INTENTS)
                p = loop._make_prediction({"intent": intent, "accepted": True,
                                          "slots": {} if value is None else {intent: value}})
                if value == "REJECT":
                    before = len(calls)
                    self.assertNotEqual(router.dispatch(p)["status"], "ok")
                    self.assertEqual(len(calls), before)
                    continue
                self.assertEqual(router.dispatch(p)["status"], "ok")
                self.assertEqual(calls[-1], (INTENTS[intent], intent, p.slot_value))
            count = len(calls)
            for p in (Prediction("OUT_OF_SCOPE", accepted=True), Prediction("COLOR", "purple", True),
                      Prediction("COLOR", None, True), Prediction("LIGHT_ON", accepted=False)):
                self.assertNotEqual(router.dispatch(p)["status"], "ok")
            self.assertEqual(len(calls), count)

    def capture(self, levels):
        clock = [0.0]
        args = SimpleNamespace(vad_frame_ms=40, command_seconds=3.0, min_command_seconds=.2,
             pre_roll_ms=200, end_silence_ms=240, speech_start_timeout=.8,
             vad_noise_seconds=.25, vad_min_speech_db=-45, vad_margin_db=10, speech_start_ms=120)
        class Stream:
            def __init__(self, **kw): self.index = 0
            def __enter__(self): return self
            def __exit__(self, *exc): pass
            def read(self, n):
                db = levels[self.index] if self.index < len(levels) else -60
                self.index += 1
                clock[0] += .04
                return np.full(n, 10**(db/20), dtype=np.float32).tobytes(), False
        class Resampler:
            def __init__(self, *a): pass
            def push(self, x): return x
        with patch.dict(sys.modules, {"sounddevice": SimpleNamespace(RawInputStream=Stream)}), \
             patch("hey_swift.audio.capture.capture_supported_rate", return_value=(16000, 640)), \
             patch("hey_swift.audio.capture.ContinuousResampler", Resampler), \
             patch.object(loop.time, "monotonic", side_effect=lambda: clock[0]):
            return loop._capture_command_vad(0, args)

    def test_noise_and_speech(self):
        for levels in ([-60]*40, [-60]*8+[-15]+[-60]*35,
                       [-60]*8+[-15,-15]+[-60]*35, [-15]+[-60]*40):
            wav, meta = self.capture(levels)
            self.assertFalse(meta["speech_started"])
            self.assertEqual(meta["reason"], "speech_start_timeout")
        for levels in ([-60]*8+[-20]*8+[-60]*20,
                       [-60]*8+[-15]+[-60]*3+[-20]*8+[-60]*20):
            wav, meta = self.capture(levels)
            self.assertTrue(meta["speech_started"])
            self.assertEqual(meta["reason"], "end_silence")
            self.assertEqual(np.count_nonzero(wav > .05), 8*640)


    def test_capture_ready_and_rate_reuse(self):
        clock = [0.0]
        active = [False]
        reads = [0]
        cue = []
        args = SimpleNamespace(vad_frame_ms=40, command_seconds=3, min_command_seconds=.2,
            pre_roll_ms=200, end_silence_ms=240, speech_start_timeout=.8,
            vad_noise_seconds=.16, vad_min_speech_db=-45, vad_margin_db=10, speech_start_ms=120)
        class Stream:
            def __init__(self, **kw): pass
            def __enter__(self): active[0] = True; return self
            def __exit__(self, *exc): active[0] = False
            def abort(self): active[0] = False
            def start(self): active[0] = True
            def read(self, n):
                self_test.assertTrue(active[0]); reads[0] += 1; clock[0] += .04
                db = -60 if reads[0] <= 4 or reads[0] > 12 else -20
                return np.full(n, 10**(db/20), dtype=np.float32).tobytes(), False
        class Resampler:
            def __init__(self, *a): pass
            def push(self, x): return x
        self_test = self
        def before():
            self.assertFalse(active[0]); self.assertEqual(reads[0], 4); cue.append(True)
        with patch.dict(sys.modules, {"sounddevice": SimpleNamespace(RawInputStream=Stream)}), \
             patch("hey_swift.audio.capture.capture_supported_rate") as probe, \
             patch("hey_swift.audio.capture.ContinuousResampler", Resampler), \
             patch.object(loop.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(loop, "_emit"):
            wav, meta = loop._capture_command_vad(0, args, rate=16000, before_ready=before)
            probe.assert_not_called()
        self.assertTrue(cue); self.assertFalse(active[0]); self.assertTrue(meta["speech_started"])
        self.assertEqual(np.count_nonzero(wav > .05), 8*640)
        self.assertLessEqual(wav.size, 8*640 + 3200 + 1920)

    def test_duck_restore_and_music_volume(self):
        class Player:
            def __init__(self): self.volume = 60; self.writes = []
            def command(self, *a, **kw):
                if a[0] == "get_property": return self.volume
                self.volume = a[-1]; self.writes.append(self.volume)
            def ctl(self, *a, **kw):
                if a[0] == "now": return {"volume": self.volume}
                self.volume = float(a[-1]); self.writes.append(self.volume)
        for active in ("offline", "spotify", None):
            for fail in (False, True):
                player = Player()
                router = SimpleNamespace(services={"music": SimpleNamespace(backend=SimpleNamespace(
                    active=active, offline=player, online=player))})
                with patch.object(loop, "_emit"):
                    try:
                        with loop._duck_music(router) as restore:
                            if active: self.assertEqual(player.volume, 0)
                            if fail: raise ValueError("capture")
                    except ValueError: self.assertTrue(fail)
                self.assertEqual(player.volume, 60)
            with patch.object(loop, "_emit"):
                with loop._duck_music(router) as restore:
                    restore(); player.volume = 70
                self.assertEqual(player.volume, 70)  # Never undo VOLUME_UP.

    def test_pipeline_lifecycle_and_no_speech(self):
        from hey_swift.integration import runtime
        from hey_swift.vcm import inference as vi
        from hey_swift.wakeword import inference as wi
        import importlib
        capture = importlib.import_module("hey_swift.audio.capture")
        playback = importlib.import_module("hey_swift.audio.playback")
        for no_speech, fail, cycles in ((False, False, 1), (False, False, 2), (True, False, 1), (False, True, 1)):
            opened = [False]; probes = []; captures = []
            args = SimpleNamespace(log_format="text", device="0", wake_config="w",vcm_config="v",
                hardware_config="h",wake_onnx="w.onnx",vcm_onnx="v.onnx",vcm_labels="l.json",
                ack="cue.wav",threshold=None,no_ack=False,live=False,hardware="mock",speak=False,
                verbose=False,wake_log_interval=2,music_duck_percent=0,playback_tail_seconds=0,
                benchmark_log=False,once=cycles==1)
            class Stream:
                def __init__(self, **kw): self_test.assertFalse(opened[0])
                def __enter__(self): opened[0] = True; return self
                def __exit__(self, *exc): opened[0] = False
                def read(self,n): return np.zeros(n,dtype=np.float32).tobytes(),False
            detector = SimpleNamespace(gate=SimpleNamespace(threshold=.3),
                                      process=lambda x:([{"prob":.9}],.9),reset=lambda:None)
            model = SimpleNamespace(predict=__import__("unittest.mock",fromlist=["Mock"]).Mock(
                return_value={"intent":"LIGHT_ON","accepted":True,"slots":{}}))
            class Service:
                def execute(self,*a,**k): return ActionResult(a[0],"ok","Lights on.")
            router = ActionRouter({"devices":Service()})
            self_test = self
            def fake_capture(device,args,rate=None,before_ready=None,stream_api=None):
                self.assertFalse(opened[0]); self.assertEqual(rate,16000)
                before_ready(); captures.append(True)
                if len(captures) == cycles: args.once = True
                if fail: raise RuntimeError("capture failed")
                return np.ones(100),{"speech_started":not no_speech,"_wake_ready_seconds":0}
            mic_info={"name":"Test USB mic","hostapi":0,"max_input_channels":1}
            fake_sd=SimpleNamespace(RawInputStream=Stream,PortAudioError=OSError,
                query_devices=lambda *a:mic_info if a else [mic_info],
                query_hostapis=lambda *a:{"name":"Test ALSA"})
            with patch.dict(sys.modules,{"sounddevice":fake_sd}), \
                 patch.object(capture,"capture_supported_rate",side_effect=lambda d:(probes.append(d) or (16000,1600))), \
                 patch.object(capture,"ContinuousResampler",return_value=SimpleNamespace(push=lambda x:x,reset=lambda:None)), \
                 patch.object(wi,"WakeWordDetector",return_value=detector), \
                 patch.object(vi,"VCMInference",return_value=model), \
                 patch.object(playback,"make_playback",return_value=object()), \
                 patch.object(runtime,"prepare_workspace"),patch.object(runtime,"build_router",return_value=router), \
                 patch.object(loop,"_capture_command_vad",side_effect=fake_capture), \
                 patch.object(loop,"_play_ack"),patch.object(loop,"_emit"):
                if fail:
                    with self.assertRaises(RuntimeError): loop.run(args)
                else: self.assertEqual(loop.run(args),0)
            self.assertFalse(opened[0]); self.assertEqual(len(probes),1); self.assertTrue(captures)
            self.assertEqual(model.predict.call_count,0 if no_speech or fail else cycles)
            self.assertTrue(router._closed)

    def test_linux_ack_and_formats(self):
        with patch.object(loop.sys,"platform","linux"), \
             patch("hey_swift.audio.speech.play_wav") as play,patch.object(loop,"_emit") as emit:
            loop._play_ack(object(),ROOT/"audio/wake_chime.wav")
            play.assert_called_once(); self.assertEqual(emit.call_args.args[0]["status"],"ok")
        event={"event":"command","capture":{"speech_started":False},
               "vcm":{"intent":"PLAY_MUSIC","confidence":.3926,"accepted":False},
               "action":{"status":"rejected","message":"Rejected"}}
        for fmt in ("text","json"):
            out=io.StringIO()
            with patch.object(loop,"LOG_FORMAT",fmt),contextlib.redirect_stdout(out): loop._emit(event)
            if fmt=="json": self.assertEqual(json.loads(out.getvalue()),event)
            else: self.assertIn("39.3%",out.getvalue()); self.assertIn("accepted: False",out.getvalue())


    def test_speech_cache(self):
        from hey_swift.audio.speech import PiperSpeech
        played=[]; synthesized=[]
        class Voice:
            def synthesize_wav(self,text,wav):
                synthesized.append(text)
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
                wav.writeframes(b"\x00\x00"*100)
        speech=PiperSpeech("unused",player=lambda p:played.append(p.read_bytes()))
        speech._voice=Voice()
        speech.speak("Lights on."); speech.speak("Lights on.")
        self.assertEqual(synthesized,["Lights on."]); self.assertEqual(len(played),2)
        self.assertEqual(played[0],played[1])
        for i in range(20): speech.speak(str(i))
        self.assertEqual(len(speech._cache),21)
        speech.close()

    def test_weather_deadline_and_backoff(self):
        import threading,time
        from hey_swift.actions.services.weather import OpenMeteoWeatherProvider
        from hey_swift.actions.services.weather_types import WeatherUnavailable
        provider=OpenMeteoWeatherProvider(fetch=lambda u:{"ok":True})
        self.assertEqual(provider._bounded_fetch(),{"ok":True})
        release=threading.Event(); calls=[]
        def blocked(u): calls.append(u); release.wait(8); return {}
        provider=OpenMeteoWeatherProvider(fetch=blocked); start=time.monotonic()
        try:
            with self.assertRaises(WeatherUnavailable): provider._bounded_fetch()
            self.assertLess(time.monotonic()-start,3.5)
            with self.assertRaises(WeatherUnavailable): provider._bounded_fetch()
            self.assertEqual(len(calls),1)
        finally: release.set()

if __name__ == "__main__": unittest.main()
