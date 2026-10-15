"""Hardware-free end-to-end safety checks, runnable in the transfer bundle."""
import hashlib
import asyncio
import json
import sys
import tempfile
import threading
import time
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hey_swift.audio.endpoint import SpeechFilter, prepare_command_audio
from hey_swift.audio.focus import AudioFocus
from hey_swift.audio.speech import PiperSpeech, SpeechService
from hey_swift.integration import e2e_loop as loop
from hey_swift.integration.router import ActionRouter, Prediction


class CaptureTests(unittest.TestCase):
    def capture(self, levels, *, overflow=False, cue_failure=False, rate=16000):
        now=[0.]; active=[False]; calls=[]
        class Stream:
            def __init__(self, samplerate, blocksize, **kwargs):
                self.rate=samplerate; self.frame=blocksize; self.i=0
            def __enter__(self): active[0]=True; return self
            def __exit__(self,*args): active[0]=False
            def abort(self): active[0]=False; calls.append("abort")
            def start(self): active[0]=True; calls.append("start")
            def read(self,n):
                now[0]+=n/self.rate
                amplitude=.001 if self.i<4 else levels[min(self.i-4,len(levels)-1)]
                self.i+=1
                return np.full(n,amplitude,dtype=np.float32).tobytes(),overflow and self.i==6
        args=types.SimpleNamespace(vad_frame_ms=40,command_seconds=5.,min_command_seconds=.5,
            speech_start_ms=120,pre_roll_ms=200,end_silence_ms=550,speech_start_timeout=3.,
            vad_noise_seconds=.16,vad_margin_db=10,vad_min_speech_db=-45)
        def cue():
            self.assertFalse(active[0]); now[0]+=.718
            if cue_failure: raise RuntimeError("cue failed")
        with patch.dict(sys.modules,{"sounddevice":types.SimpleNamespace(RawInputStream=Stream)}), \
             patch.object(loop.time,"monotonic",side_effect=lambda:now[0]),patch.object(loop,"_emit"):
            try: return loop._capture_command_vad(None,args,rate=rate,before_ready=cue)
            finally: self.assertFalse(active[0])

    def test_immediate_onset_preserved_without_duplicate_frames(self):
        audio,meta=self.capture([.05]*8+[.001]*30)
        self.assertTrue(meta["speech_started"])
        self.assertEqual(np.count_nonzero(audio==np.float32(.05)),8*640)
        self.assertEqual(meta["reason"],"end_silence")
        self.assertLessEqual(meta["endpoint_wait_seconds"],.6)

    def test_delayed_speech_gets_its_own_deadline(self):
        audio,meta=self.capture([.001]*50+[.05]*8+[.001]*30)
        self.assertTrue(meta["speech_started"])
        self.assertGreater(meta["speech_wait_seconds"],1.9)
        self.assertGreater(len(audio),0)

    def test_short_click_and_silence_do_not_start_command(self):
        for levels in ([.001]*100,[.05]+[.001]*100):
            _,meta=self.capture(levels)
            self.assertFalse(meta["speech_started"])

    def test_internal_pause_is_preserved(self):
        audio,meta=self.capture([.05]*6+[.001]*6+[.05]*6+[.001]*30)
        self.assertEqual(np.count_nonzero(audio==np.float32(.05)),12*640)
        self.assertEqual(meta["reason"],"end_silence")

    def test_quiet_above_floor_speech_is_preserved(self):
        _,meta=self.capture([.007]*8+[.001]*30)
        self.assertTrue(meta["speech_started"])

    def test_overflow_fails_closed(self):
        audio,meta=self.capture([.05]*8+[.001]*30,overflow=True)
        self.assertEqual(prepare_command_audio(audio,meta,2.)[1],"capture_overflow")

    def test_two_second_fit_does_not_remove_detected_speech(self):
        audio,meta=self.capture([.001]*4+[.05]*48+[.001]*30)
        fitted,reason=prepare_command_audio(audio,meta,2.)
        self.assertIsNone(reason); self.assertEqual(len(fitted),32000)
        self.assertEqual(np.count_nonzero(fitted==np.float32(.05)),48*640)

    def test_long_and_incomplete_commands_do_not_reach_actions(self):
        for levels in ([.05]*70+[.001]*30,[.05]*150):
            audio,meta=self.capture(levels)
            fitted,reason=prepare_command_audio(audio,meta,2.)
            self.assertIsNone(fitted)
            service=types.SimpleNamespace(execute=lambda *a: self.fail("Unsafe action"))
            router=ActionRouter({"devices":service})
            result=router.dispatch(Prediction("LIGHT_ON",accepted=False),rejection_reason=reason)
            self.assertEqual(result["status"],"rejected")
            self.assertEqual(result["message"],"Please say a shorter command.")

    def test_failed_cue_closes_capture(self):
        with self.assertRaises(RuntimeError): self.capture([.05]*8,cue_failure=True)

    def test_pi_native_rate_resamples_and_endpoints(self):
        audio,meta=self.capture([.05]*8+[.001]*30,rate=48000)
        self.assertTrue(meta["speech_started"])
        self.assertEqual(meta["reason"],"end_silence")
        self.assertTrue(np.isfinite(audio).all())
        self.assertLess(len(audio),16000)


class FocusAndSpeechTests(unittest.TestCase):
    def test_scheduled_speech_waits_until_command_capture_finishes(self):
        focus=AudioFocus(tail_seconds=0); started=threading.Event(); played=threading.Event()
        backend=types.SimpleNamespace(speak=lambda text:played.set())
        speech=SpeechService(backend,before_playback=focus.before_playback,after_playback=focus.after_playback)
        def worker(): started.set(); speech.speak("Your timer is finished.")
        with focus.capture():
            thread=threading.Thread(target=worker); thread.start(); started.wait(1)
            self.assertFalse(played.wait(.03))
        thread.join(1); self.assertFalse(thread.is_alive()); self.assertTrue(played.is_set())

    def test_playback_error_releases_focus_and_applies_tail_guard(self):
        focus=AudioFocus(tail_seconds=.02)
        def fail(text): self.assertTrue(focus.suppress_input()); raise RuntimeError("sink failed")
        speech=SpeechService(types.SimpleNamespace(speak=fail),before_playback=focus.before_playback,after_playback=focus.after_playback)
        with self.assertRaises(RuntimeError): speech.speak("Lights on.")
        self.assertTrue(focus.suppress_input())
        with focus.capture(): self.assertFalse(focus.suppress_input())

    def test_action_result_is_emitted_before_speech(self):
        from hey_swift.actions.services.information import ActionResult
        events=[]
        service=types.SimpleNamespace(execute=lambda *a:ActionResult("LIGHT_ON","ok","Lights on."))
        speech=types.SimpleNamespace(speak=lambda text:events.append("speech"))
        router=ActionRouter({"devices":service},speech=speech)
        router.on_action=lambda *args:events.append("result")
        router.dispatch(Prediction("LIGHT_ON",accepted=True))
        self.assertEqual(events,["result","speech"])

    def test_precomputed_phrase_avoids_live_synthesis(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); model=root/"voice.onnx"; model.write_bytes(b"voice")
            Path(str(model)+".json").write_text("{}")
            audio=root/"cached.wav"
            with wave.open(str(audio),"wb") as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000); wav.writeframes(b"\0\0"*160)
            (root/"manifest.json").write_text(json.dumps({"voice_sha256":hashlib.sha256(b"voice").hexdigest(),
                "phrases":{"Lights on.":{"file":"cached.wav","sha256":hashlib.sha256(audio.read_bytes()).hexdigest()}}}))
            played=[]; speech=PiperSpeech(model,player=played.append,phrases_dir=root)
            speech._voice=types.SimpleNamespace(synthesize_wav=lambda *args:self.fail("Unexpected synthesis"))
            timing=speech.speak("Lights on."); self.assertTrue(timing["cached"])
            self.assertEqual(played,[audio]); speech.close()

    def test_webrtc_framing_and_energy_fallback(self):
        frames=[]
        vad=types.SimpleNamespace(Vad=lambda mode:types.SimpleNamespace(is_speech=lambda pcm,sr:frames.append((len(pcm),sr)) or False))
        with patch.dict(sys.modules,{"webrtcvad":vad}):
            speech=SpeechFilter("webrtc")
            self.assertFalse(speech.voiced(np.ones(640,dtype=np.float32)*.01))
        self.assertEqual(frames,[(320,16000)]*4)
        self.assertTrue(SpeechFilter().voiced(np.zeros(640,dtype=np.float32)))

    def test_queued_completion_cannot_deadlock_foreground_speech(self):
        focus=AudioFocus(tail_seconds=0); messages=[]
        speech=SpeechService(types.SimpleNamespace(speak=messages.append))
        speech.set_playback_hooks(focus.before_playback,focus.after_playback,lock=focus.lock)
        with focus.capture():
            speech.enqueue("Your timer is finished.")
            time.sleep(.02)
            speech.speak("Lights on.")
            self.assertEqual(messages,["Lights on."])
        speech._notifications.join()
        self.assertEqual(messages,["Lights on.","Your timer is finished."])
        speech.close(); self.assertFalse(speech._worker.is_alive())

    def test_notification_failure_does_not_disable_later_speech(self):
        messages=[]
        def speak(text):
            if text=="failed": raise RuntimeError("Audio unavailable")
            messages.append(text)
        speech=SpeechService(types.SimpleNamespace(speak=speak))
        speech.enqueue("failed"); speech._notifications.join()
        speech.enqueue("working"); speech._notifications.join()
        self.assertEqual(messages,["working"]); self.assertEqual(len(speech.notification_errors),1)
        speech.close()

    def test_timer_worker_can_finish_and_cancel_during_audio_ownership(self):
        from hey_swift.integration.runtime import ScheduledDevices
        from hey_swift.hardware.devices import MockHardware
        focus=AudioFocus(tail_seconds=0); messages=[]
        speech=SpeechService(types.SimpleNamespace(speak=messages.append))
        speech.set_playback_hooks(focus.before_playback,focus.after_playback,lock=focus.lock)
        devices=ScheduledDevices(MockHardware(),speech=speech)
        with focus.capture():
            devices._thread=threading.Thread(target=devices._say,args=("Your timer is finished.",))
            devices._thread.start(); devices._thread.join(1)
            self.assertFalse(devices._thread.is_alive())
            devices.cancel()
            self.assertEqual(messages,[])
        speech._notifications.join()
        self.assertEqual(messages,["Your timer is finished."])
        devices.close(); speech.close()

    def test_invalid_phrase_manifest_does_not_prevent_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/"manifest.json").write_text("not json")
            speech=PiperSpeech(root/"missing.onnx",phrases_dir=root)
            self.assertEqual(speech._precomputed,{})
            from hey_swift.actions.services.information import ActionResult
            action=types.SimpleNamespace(execute=lambda *args:ActionResult("LIGHT_ON","ok","Lights on."))
            router=ActionRouter({"devices":action},speech=SpeechService(speech))
            result=router.dispatch(Prediction("LIGHT_ON",accepted=True))
            self.assertEqual(result["status"],"ok")
            self.assertEqual(result["speech"]["status"],"unavailable")


class OfflineTests(unittest.TestCase):
    def test_call_timeout_is_bounded_and_unknown_after_dial(self):
        import subprocess
        from hey_swift.actions.services.calling import BluetoothCallingBackend
        from hey_swift.integration.runtime import CallingService
        calls=[]
        def runner(command,**kwargs):
            calls.append(command); self.assertLessEqual(kwargs["timeout"],.02)
            method=command[7]
            if method=="Dial": raise subprocess.TimeoutExpired(command,.02)
            body={"data":[{"/gateway":{}}]} if method=="GetModems" else {"data":[[]]}
            return types.SimpleNamespace(stdout=json.dumps(body))
        service=CallingService("+12025550123",backend=BluetoothCallingBackend(runner=runner,timeout=.02))
        result=service.execute("CALL")
        self.assertEqual(result.status,"unknown")
        self.assertEqual(sum(command[7]=="Dial" for command in calls),1)

    def test_message_total_budget_and_submission_uncertainty(self):
        from hey_swift.actions.services.messaging import BluetoothMessagingBackend, MessagingService
        for stage in ("connect","PushMessage"):
            calls=[]; disconnected=[]
            class Bus:
                async def connect(self):
                    if stage=="connect": await asyncio.sleep(1)
                    return self
                def add_message_handler(self,callback): pass
                def remove_message_handler(self,callback): pass
                def disconnect(self): disconnected.append(True)
                async def call(self,message):
                    calls.append(message.member)
                    if message.member=="PushMessage": await asyncio.sleep(1)
                    body=[]
                    if message.member=="CreateSession": body=["/session"]
                    if message.member=="Get": body=[types.SimpleNamespace(value=["SMS_GSM"])]
                    return types.SimpleNamespace(message_type="reply",body=body)
            dbus=types.ModuleType("dbus_next")
            dbus.Message=lambda **kw:types.SimpleNamespace(**kw)
            dbus.MessageType=types.SimpleNamespace(ERROR="error",SIGNAL="signal")
            dbus.Variant=lambda *a:types.SimpleNamespace(value=a[1])
            aio=types.ModuleType("dbus_next.aio"); aio.MessageBus=Bus
            with patch.dict(sys.modules,{"dbus_next":dbus,"dbus_next.aio":aio}):
                backend=BluetoothMessagingBackend("AA:BB:CC:DD:EE:FF",timeout=45,total_timeout=.02)
                began=time.monotonic()
                result=MessagingService(backend,"+12025550123").execute()
            self.assertLess(time.monotonic()-began,.15)
            self.assertEqual(result.status,"unavailable" if stage=="connect" else "unknown")
            self.assertLessEqual(calls.count("PushMessage"),1)
            if stage=="PushMessage": self.assertTrue(disconnected)

    def test_message_session_retries_obey_budget_before_submission(self):
        from hey_swift.actions.services.messaging import create_map_session, MessagingUnavailable
        calls=[]
        async def unavailable(*args):
            calls.append(True)
            raise MessagingUnavailable(diagnostic={"dbus_error":"org.freedesktop.DBus.Error.UnknownMethod","method":"CreateSession"})
        began=time.monotonic()
        with self.assertRaises(asyncio.TimeoutError):
            asyncio.run(create_map_session(unavailable,"AA:BB:CC:DD:EE:FF",lambda *a:None,retry_budget=lambda:.01))
        self.assertLess(time.monotonic()-began,.1); self.assertEqual(len(calls),1)

    def test_offline_router_demonstrates_local_actions_without_internet(self):
        from hey_swift.integration import runtime
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(runtime,"ROOT",root),patch.object(Path,"cwd",return_value=root), \
                 patch.object(runtime.IntegrationSoloist,"execute",side_effect=AssertionError("Spotify called offline")), \
                 patch.object(runtime.MpvBackend,"execute",return_value={"backend":"offline","playback_confirmed":True}), \
                 patch("hey_swift.actions.services.weather.urlopen",side_effect=AssertionError("Internet called offline")):
                with runtime.build_router(live=True,hardware="mock",offline=True) as router:
                    for intent,slot in (("TIME",None),("LIGHT_ON",None),("COLOR","Blue"),
                        ("TEMPERATURE","18 degrees"),("BRIGHTNESS","20 percent"),
                        ("TIMER","10 seconds"),("CREATE_REMINDER","Drink water"),("LIST_REMINDERS",None),
                        ("PLAY_MUSIC",None),("PAUSE",None),("VOLUME_DOWN",None),("STOP",None)):
                        result=router.dispatch(Prediction(intent,slot,True))
                        self.assertEqual(result["status"],"ok",msg=str(result))
                    result=router.dispatch(Prediction("WEATHER",accepted=True))
                    self.assertEqual(result["status"],"unavailable")
                    self.assertNotIn("temperature_c",result["data"])

    def test_hung_weather_does_not_repeat_blocking_wait(self):
        from hey_swift.actions.services.weather import OpenMeteoWeatherProvider
        from hey_swift.actions.services.weather_types import WeatherUnavailable
        unblock=threading.Event(); calls=[]
        def fetch(url): calls.append(True); unblock.wait(1); raise OSError("No network")
        provider=OpenMeteoWeatherProvider(fetch=fetch,wait_seconds=.01)
        try:
            with self.assertRaises(WeatherUnavailable): provider.current_weather()
            began=time.monotonic()
            for _ in range(5):
                with self.assertRaises(WeatherUnavailable): provider.current_weather()
            self.assertLess(time.monotonic()-began,.05); self.assertEqual(len(calls),1)
        finally:
            unblock.set()
            if provider._pending: provider._pending[0].join(1)

    def test_online_setup_failure_falls_back_without_fake_success(self):
        from hey_swift.actions.services.music import AutoMusicBackend, MusicUnavailable, MusicStartUncertain
        offline=types.SimpleNamespace(execute=lambda intent:{"backend":"offline"})
        online=types.SimpleNamespace(execute=lambda intent:(_ for _ in ()).throw(MusicUnavailable("offline")))
        router=AutoMusicBackend(online,offline,reachable=lambda:True,state_path=None)
        result=router.execute("PLAY_MUSIC"); self.assertTrue(result["fallback"])
        online.execute=lambda intent:(_ for _ in ()).throw(MusicStartUncertain("uncertain"))
        router.active=None
        with self.assertRaises(MusicStartUncertain): router.execute("PLAY_MUSIC")

    def test_soloist_control_deadline_is_bounded(self):
        from hey_swift.actions.services.music import SoloistBackend, MusicUnavailable
        timeouts=[]
        def runner(command,**kwargs): timeouts.append(kwargs["timeout"]); raise OSError("offline")
        backend=SoloistBackend(runner=runner,control_timeout=.2,operation_timeout=.4,readiness_attempts=1)
        with self.assertRaises(MusicUnavailable): backend.execute("PLAY_MUSIC")
        self.assertEqual(len(timeouts),1); self.assertLessEqual(timeouts[0],.2)


if __name__=="__main__": unittest.main()
