"""Focused v6 regression checks; no GPIO, calls, SMS, network or audio devices."""
import io
import json
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hey_swift.audio.recovery import StableMicrophone, MicrophoneUnavailable, MicrophoneRecoveryExhausted
from hey_swift.audio.cue import prepared_cue
from hey_swift.audio.endpoint import prepare_command_audio
from hey_swift.audio.pipewire_capture import PipeWireCapture, PipeWireInputStream, PipeWireCaptureError
from hey_swift.integration.call_guard import wait_for_call_end, CallGuardUnavailable
from hey_swift.integration import e2e_loop as loop
from hey_swift.actions.services.music import SoloistBackend, AutoMusicBackend, MusicUnavailable, MusicStartUncertain, MusicService
from hey_swift.actions.services.device_actions import DeviceActions
from hey_swift.hardware.devices import MockHardware


class MicrophoneTests(unittest.TestCase):
    def api(self):
        rows=[{"name":"USB microphone","hostapi":0,"max_input_channels":1}]
        def devices(*args): return rows[0] if args else rows
        return types.SimpleNamespace(query_devices=devices,query_hostapis=lambda n:{"name":"ALSA"},PortAudioError=OSError),rows

    def test_reindex_follows_identity_not_old_index(self):
        sd,rows=self.api(); mic=StableMicrophone(sd,0)
        rows.insert(0,{"name":"Wrong microphone","hostapi":0,"max_input_channels":1})
        self.assertEqual(mic.resolve(),1)

    def test_missing_or_duplicate_never_uses_another_input(self):
        sd,rows=self.api(); mic=StableMicrophone(sd,0)
        rows.append(dict(rows[0]))
        with self.assertRaises(MicrophoneUnavailable): mic.resolve()
        rows[:]=[{"name":"Wrong microphone","hostapi":0,"max_input_channels":1}]
        with self.assertRaises(MicrophoneUnavailable): mic.resolve()

    def test_retries_back_off_and_exhaust(self):
        sd,_=self.api(); events=[]; mic=StableMicrophone(sd,0,attempts=3,emit=events.append)
        with patch("hey_swift.audio.recovery.time.sleep") as sleep:
            for _ in range(3): mic.recover(OSError("Device unavailable -9985"))
            with self.assertRaises(MicrophoneRecoveryExhausted): mic.recover(OSError("busy"))
        self.assertEqual([c.args[0] for c in sleep.call_args_list],[1,2,4])
        self.assertEqual(events[-1]["status"],"exhausted")

    def test_wake_stream_recovers_without_action_or_wrong_input(self):
        sd,_=self.api(); attempts=[]; closed=[]; events=[]
        class Stream:
            def __init__(self,**kw): attempts.append(kw["device"])
            def __enter__(self):
                if len(attempts)==1: raise OSError("Device unavailable -9985")
                return self
            def __exit__(self,*args): closed.append(True)
            def read(self,n): return np.zeros(n,dtype=np.float32).tobytes(),False
        sd.RawInputStream=Stream; mic=StableMicrophone(sd,0,emit=events.append)
        detector=types.SimpleNamespace(process=lambda data:([{"prob":.9}],.9),reset=lambda:None)
        resampler=types.SimpleNamespace(push=lambda data:data,reset=lambda:None)
        with patch("hey_swift.audio.recovery.time.sleep"),patch.object(loop,"_emit"):
            loop._listen_for_wake(sd,mic,48000,4800,detector,resampler,
                types.SimpleNamespace(suppress_input=lambda:False),types.SimpleNamespace(wake_log_interval=2,verbose=False))
        self.assertEqual(attempts,[0,0]); self.assertEqual(closed,[True]); self.assertEqual(len(events),1)


class CaptureAndCueTests(unittest.TestCase):
    def test_endpoint_wait_is_not_model_duration(self):
        wav=np.zeros(34000,dtype=np.float32)
        fitted,reason=prepare_command_audio(wav,{"reason":"end_silence","capture_seconds":2.932,
            "speech_start_sample":3000,"speech_end_sample":31000},2.)
        self.assertIsNone(reason); self.assertEqual(len(fitted),32000)

    def test_true_long_speech_remains_rejected(self):
        fitted,reason=prepare_command_audio(np.zeros(47000,dtype=np.float32),
            {"reason":"end_silence","speech_start_sample":1000,"speech_end_sample":46000},2.)
        self.assertIsNone(fitted); self.assertEqual(reason,"command_too_long")

    def test_cue_gain_is_peak_limited_and_original_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"local_plus3.wav"; sf.write(path,np.array([.9,-.9]*160),16000,subtype="FLOAT")
            original=path.read_bytes()
            with prepared_cue(path,6) as (output,info):
                samples,_=sf.read(output)
                self.assertLessEqual(np.max(np.abs(samples)),.981)
                self.assertTrue(info["peak_limited"])
                self.assertLess(info["applied_gain_db"],6)
            self.assertEqual(path.read_bytes(),original); self.assertFalse(output.exists())

    def test_zero_gain_preserves_pi_asset_and_low_peak_can_gain(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"cue.wav"; sf.write(path,np.ones(1000)*.1,16000)
            with prepared_cue(path,0) as (output,info): self.assertEqual(output,path)
            with prepared_cue(path,3) as (output,info):
                self.assertFalse(info["peak_limited"]); self.assertAlmostEqual(info["applied_gain_db"],3)


class CallGuardTests(unittest.TestCase):
    def test_call_guard_waits_for_idle_and_hangup_tail_without_dial(self):
        now=[0.]; statuses=iter([True,True,False,False]); events=[]
        service=types.SimpleNamespace(backend=types.SimpleNamespace(has_active_calls=lambda gateway:next(statuses)))
        with patch("hey_swift.integration.call_guard.time.monotonic",side_effect=lambda:now[0]), \
             patch("hey_swift.integration.call_guard.time.sleep",side_effect=lambda seconds:now.__setitem__(0,now[0]+seconds)):
            wait_for_call_end(service,{"data":{"gateway":"/gateway"}},emit=events.append)
        self.assertEqual(now[0],4.5); self.assertEqual(events[-1]["status"],"resuming")

    def test_call_state_error_or_timeout_never_resumes(self):
        for broken in (False,True):
            now=[0.]; events=[]
            def status(gateway):
                if broken: raise OSError("Bluetooth unavailable")
                return True
            service=types.SimpleNamespace(backend=types.SimpleNamespace(has_active_calls=status))
            with patch("hey_swift.integration.call_guard.time.monotonic",side_effect=lambda:now[0]), \
                 patch("hey_swift.integration.call_guard.time.sleep",side_effect=lambda seconds:now.__setitem__(0,now[0]+seconds)):
                with self.assertRaises(CallGuardUnavailable):
                    wait_for_call_end(service,{"data":{"gateway":"/gateway"}},emit=events.append,maximum=5)
            self.assertNotIn("resuming",[e["status"] for e in events])

    def test_paused_notifications_do_not_speak_into_uncertain_call(self):
        from hey_swift.audio.speech import SpeechService
        messages=[]; speech=SpeechService(types.SimpleNamespace(speak=messages.append))
        speech.pause_notifications(); speech.enqueue("Timer finished."); speech._notifications.join()
        self.assertEqual(messages,[])
        speech.pause_notifications(False); speech.enqueue("Ready."); speech._notifications.join()
        self.assertEqual(messages,["Ready."]); speech.close()


class TimerTests(unittest.TestCase):
    def test_all_timer_slots_count_down_independently_of_wake_loop(self):
        for value,seconds in (("10 seconds",10),("30 seconds",30),("1 minute",60)):
            now=[0.]; hardware=MockHardware(); service=DeviceActions(hardware)
            class DummyThread:
                def __init__(self,**kw): pass
                def start(self): pass
            with patch("hey_swift.actions.services.device_actions.threading.Thread",DummyThread):
                result=service.execute("TIMER",value)
            self.assertEqual(result.status,"ok")
            class Stop:
                def wait(self,seconds): now[0]+=1; return False
                def is_set(self): return False
            service._stop=Stop()
            with patch("hey_swift.actions.services.device_actions.time.monotonic",side_effect=lambda:now[0]):
                service._job("TIMER",seconds,None)
            digits=[e[1] for e in hardware.events if e[0]=="display"]
            self.assertEqual(digits[0],"0100" if seconds==60 else f"00{seconds:02d}")
            self.assertIn("0059" if seconds==60 else f"00{seconds-1:02d}",digits)
            self.assertEqual(digits[-1],"0000"); self.assertEqual(service.events[-1].status,"completed")

    def test_display_override_cancels_old_job_but_bad_slot_does_not(self):
        hardware=MockHardware(); service=DeviceActions(hardware); cancelled=[]
        class Job:
            def is_alive(self): return True
            def join(self): cancelled.append(True)
        service._thread=Job()
        self.assertEqual(service.execute("TIMER","unsupported").status,"invalid_slot")
        self.assertFalse(cancelled)
        self.assertEqual(service.execute("TEMPERATURE","22 degrees").status,"ok")
        self.assertEqual(cancelled,[True]); self.assertEqual(hardware.events[-1][1],"  22")

    def test_background_display_failure_is_reported_once(self):
        hardware=MockHardware(); hardware.display=lambda *args:(_ for _ in ()).throw(OSError("TM1637 no ACK"))
        service=DeviceActions(hardware)
        service._stop=types.SimpleNamespace(wait=lambda seconds:False,is_set=lambda:False)
        service._job("TIMER",10,None)
        with patch.object(loop,"_emit") as emit:
            router=types.SimpleNamespace(services={"devices":service})
            cursor=loop._drain_scheduled_events(router,0)
            self.assertEqual(loop._drain_scheduled_events(router,cursor),cursor)
        self.assertEqual(emit.call_count,1)
        self.assertEqual(emit.call_args.args[0]["data"]["error"],"TM1637 no ACK")


class SpotifyTests(unittest.TestCase):
    def backend(self):
        return SoloistBackend(readiness_attempts=8,max_volume_percent=25)

    def test_slow_activation_observed_before_play_and_saved_login_reused(self):
        backend=self.backend(); events=[]; reads=[0]
        def ctl(*args,**kwargs):
            events.append(args)
            if args==("now",):
                reads[0]+=1
                return {"is_active":reads[0]>=4,"status":"playing" if ("play",) in events else "paused","volume":25}
        backend.ctl=ctl
        with patch("hey_swift.actions.services.music.time.sleep"):
            self.assertTrue(backend.execute("PLAY_MUSIC")["playback_confirmed"])
        self.assertEqual(events.count(("activate",)),1)
        self.assertLess(events.index(("activate",)),events.index(("play",)))
        self.assertEqual(events[:5],[("now",),("activate",),("now",),("now",),("now",)])

    def test_unknown_play_acknowledgement_blocks_offline_fallback(self):
        backend=self.backend(); offline_calls=[]
        def ctl(*args,**kwargs):
            if args and args[0]=="play": raise MusicUnavailable("control timeout")
            return {"is_active":True,"status":"paused"}
        backend.ctl=ctl
        offline=types.SimpleNamespace(execute=lambda intent:offline_calls.append(intent))
        auto=AutoMusicBackend(backend,offline,reachable=lambda:True,state_path=None)
        with self.assertRaises(MusicStartUncertain): auto.execute("PLAY_MUSIC")
        self.assertEqual(offline_calls,[])

    def test_unconfirmed_pause_does_not_allow_second_player(self):
        backend=self.backend(); backend.readiness_attempts=2
        backend.ctl=lambda *a,**k:{"is_active":False,"status":"playing"}
        with patch("hey_swift.actions.services.music.time.sleep"):
            with self.assertRaises(MusicStartUncertain): backend.execute("PLAY_MUSIC")

    def test_limit_response_explains_actual_cap(self):
        backend=types.SimpleNamespace(execute=lambda intent:{"volume_limited":True,"listening_limit_percent":25})
        result=MusicService(backend).execute("VOLUME_UP")
        self.assertIn("25 percent",result.message); self.assertIn("hear you",result.message)

    def test_existing_spotify_must_confirm_pause_before_offline_switch(self):
        backend=self.backend(); offline_calls=[]
        def offline_execute(intent):
            offline_calls.append(intent)
            return {"backend":"offline"}
        offline=types.SimpleNamespace(execute=offline_execute)
        auto=AutoMusicBackend(backend,offline,reachable=lambda:False,state_path=None)
        auto.active="spotify"
        with patch.object(backend,"_pause_before_fallback",side_effect=MusicUnavailable("still playing")), \
             patch.object(backend,"is_running",return_value=True):
            with self.assertRaises(MusicUnavailable): auto.execute("PLAY_MUSIC")
        self.assertEqual(offline_calls,[])
        with patch.object(backend,"_pause_before_fallback") as pause:
            auto.execute("PLAY_MUSIC")
        pause.assert_called_once(); self.assertEqual(offline_calls,["PLAY_MUSIC"])


class PipeWireTests(unittest.TestCase):
    def test_only_exact_usb_source_can_be_selected(self):
        api=PipeWireCapture("usb-mic-serial")
        wrong=json.dumps([{"info":{"props":{"node.name":"wrong","media.class":"Audio/Source"}}}])
        right=json.dumps([{"info":{"props":{"node.name":"usb-mic-serial","media.class":"Audio/Source"}}}])
        with patch("hey_swift.audio.pipewire_capture.subprocess.run",return_value=types.SimpleNamespace(stdout=wrong)):
            with self.assertRaises(PipeWireCaptureError): api.query_devices()
        with patch("hey_swift.audio.pipewire_capture.subprocess.run",return_value=types.SimpleNamespace(stdout=right)):
            self.assertEqual(api.query_devices(0,"input")["name"],"usb-mic-serial")

    def test_pipewire_partial_reads_and_cue_discard(self):
        class Child:
            def __init__(self): self.stdout=io.BytesIO(b"\0"*128); self.dead=False
            def poll(self): return 0 if self.dead else None
            def terminate(self): self.dead=True
            def wait(self,**kwargs): return 0
        child=Child()
        with patch("hey_swift.audio.pipewire_capture.subprocess.Popen",return_value=child) as launch:
            with PipeWireInputStream("usb-mic",samplerate=48000,blocksize=16,dtype="float32",channels=1) as stream:
                data,overflow=stream.read(16)
                self.assertEqual(len(data),64)
                stream.abort(); self.assertTrue(stream._queue.empty())
                stream.start(); self.assertFalse(stream._paused)
            self.assertFalse(stream.thread.is_alive())
            self.assertIn("usb-mic",launch.call_args.args[0]); self.assertTrue(child.dead)


if __name__=="__main__": unittest.main()
