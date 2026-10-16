"""Construct bundled services with all writable paths inside integration/."""
from pathlib import Path
import os
import tempfile
import threading

from hey_swift.actions.services.information import ActionResult, InformationService
from hey_swift.actions.services.calling import BluetoothCallingBackend, CallingUnavailable, CallingUncertain
from hey_swift.actions.services.device_actions import DeviceActions
from hey_swift.actions.services.messaging import BluetoothMessagingBackend, MessagingService
from hey_swift.actions.services.music import (MusicService, MpvBackend, SoloistBackend,
                           AutoMusicBackend, DEFAULT_SPOTIFY_URI)
from hey_swift.actions.services.reminders import ReminderService, JsonReminderStore
from hey_swift.actions.services.weather import OpenMeteoWeatherProvider
from hey_swift.audio.speech import PiperSpeech, SpeechService
from hey_swift.hardware.devices import MockHardware, PiHardware
from hey_swift.integration.router import ActionRouter, GROUPS
from hey_swift.integration.config import communication_values, runtime_root, spotify_paths

ROOT = runtime_root()


def prepare_workspace():
    """Call once before starting runtime threads; this runner owns process cwd.

    Legacy mpv logs are cwd-relative. Piper/OBEX use Python's global tempdir.
    Run in a dedicated process, not inside an unrelated application's process.
    """
    ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name in ("state", "logs", "tmp"):
        (ROOT / name).mkdir(exist_ok=True, mode=0o700)
    os.chdir(ROOT)
    tempfile.tempdir = str(ROOT / "tmp")
    # Preserve native backends' shell environment, matching standalone startup.
    # Python tempfile already contains Piper/OBEX outputs; players receive
    # explicit writable socket/log/data/cache paths below.


class PreviewService:
    def execute(self, intent, slot_value=None, *, before_dial=None):
        return ActionResult(intent, "dry_run", "Command validated. No action was executed.",
                            {"slot_value": slot_value})


class CallingService:
    def __init__(self, number, backend=None):
        self.number = number
        self.backend = backend or BluetoothCallingBackend(timeout=4)

    def execute(self, intent, slot_value=None, *, before_dial=None):
        try:
            return ActionResult(**self.backend.dial(self.number, before_dial=before_dial))
        except CallingUncertain as exc:
            return ActionResult(intent, "unknown", "Check your phone before calling again.",
                                {"gateway": getattr(exc, "gateway", None)})
        except (CallingUnavailable, ValueError):
            return ActionResult(intent, "unavailable", "Calling is unavailable. Check Bluetooth.")


class ConfiguredMessaging:
    def __init__(self, address, number):
        self.address, self.number = address, number

    def execute(self, intent, slot_value=None):
        try:
            service = MessagingService(BluetoothMessagingBackend(self.address, timeout=5, total_timeout=15), self.number)
        except ValueError:
            return ActionResult(intent, "unavailable", "Messaging is unavailable. Configure the paired device and recipient.")
        return service.execute(intent, slot_value)


class ConfirmedCommunication:
    """Optional per-action terminal approval, preserving call-state access."""
    def __init__(self, service):
        self.service = service

    @property
    def backend(self):
        return self.service.backend

    def execute(self, intent, slot_value=None, **kwargs):
        import sys
        if not sys.stdin.isatty() or input(f"Type {intent} to authorize this real action: ").strip() != intent:
            return ActionResult(intent, "cancelled", "Communication was not authorized.")
        return self.service.execute(intent, slot_value, **kwargs)


class ScheduledDevices(DeviceActions):
    """Router speaks command results; workers speak completion through same TTS."""
    def _say(self, message):
        if threading.current_thread() is self._thread and self.speech:
            try:
                self.speech.enqueue(message)
            except Exception:
                # Preserve completion even when speech fails.
                self.events.append(ActionResult("SPEECH", "unavailable", "Unable to play scheduled response."))

    def close(self):
        # Explicitly blank the display on integration shutdown.
        try:
            self.cancel()
        finally:
            super().close()


class ManagedMusic(MusicService):
    def close(self):
        # Stop the backend selected by this router; daemons remain reusable.
        if self.backend.active is not None:
            self.execute("STOP")


class IntegrationSoloist(SoloistBackend):
    """Standalone startup/control unchanged, with sanitized fallback diagnostics."""
    def __init__(self, *args, profile="bundled", **kwargs):
        super().__init__(*args, **kwargs)
        self.profile = profile
        self.last_error = None

    def execute(self, intent):
        if intent == "PLAY_MUSIC":
            self.last_error = None
        try:
            return {**super().execute(intent), "spotify_config": self.profile}
        except Exception as exc:
            # Preserve only known setup messages, never subprocess output.
            from hey_swift.actions.services.music import MusicUnavailable
            if isinstance(exc, MusicUnavailable):
                self.last_error = str(exc)
            raise


class IntegrationAutoMusic(AutoMusicBackend):
    def __init__(self, *args, **kwargs):
        # The standalone --backend soloist command attempts the real backend.
        # An unrelated website probe must not skip a working Spotify session.
        kwargs.setdefault("reachable", lambda: True)
        super().__init__(*args, **kwargs)

    def execute(self, intent):
        data = super().execute(intent)
        if intent == "PLAY_MUSIC" and data.get("fallback") and self.online.last_error:
            data["spotify_diagnostic"] = {
                "message": self.online.last_error,
                "phase": self.online.phase,
                "profile": self.online.profile,
                "hint": "Check the selected Soloist runtime's login status.",
            }
        return data


def _common_speech_phrases(music_volume_limit=25):
    from itertools import permutations
    from hey_swift.actions.services.device_actions import VALUES
    from hey_swift.actions.services.music import RESPONSES
    from hey_swift.actions.services.reminders import VALUES as REMINDER_VALUES

    phrases = [
        "Sorry, I didn't understand that command.",
        "That command is not supported.",
        "This action is not configured.",
        "The action could not be completed. Please check its setup.",
        "Music is at the listening volume limit.",
        f"Music is limited to {music_volume_limit} percent so I can hear you.",
        "Spotify is unavailable. Playing your offline music.",
        "Live weather is unavailable. Please check the internet.",
        "Please say a shorter command.", "Please repeat your command.",
        "I didn't hear a command.", "Your timer is finished.", "Your alarm is ringing.",
        "That value isn't supported in this demo.",
        "Placing your call.",
        "Calling is unavailable. Check Bluetooth.",
        "Check your phone before calling again.",
        "Weather is unavailable. No weather provider is configured.",
        "You have no reminders yet.",
        "Lights on.", "Lights off.",
        *RESPONSES.values(),
    ]
    phrases.extend(f"Timer set for {value}." for value in VALUES["TIMER"])
    phrases.extend(f"Alarm set for {value}." for value in VALUES["ALARM"])
    phrases.extend(f"Temperature set to {value}." for value in VALUES["TEMPERATURE"])
    phrases.extend(f"Brightness set to {value}." for value in VALUES["BRIGHTNESS"])
    phrases.extend(f"Lights set to {value.lower()}." for value in VALUES["COLOR"])
    phrases.extend(f"Your reminder to {value.lower()} has been saved."
                   for value in REMINDER_VALUES)
    reminder_names = tuple(value.lower() for value in REMINDER_VALUES.values())
    phrases.extend(f"You already have a reminder to {name}." for name in reminder_names)
    for count in range(1, len(reminder_names) + 1):
        for ordered in permutations(reminder_names, count):
            if count == 1:
                phrase = f"Your reminder is to {ordered[0]}."
            else:
                phrase = "Your reminders are to " + ", ".join(ordered[:-1])
                phrase += f" and {ordered[-1]}."
            phrases.append(phrase)
    return tuple(dict.fromkeys(phrases))


def build_router(*, live=False, hardware="mock", speak=False,
                 music_volume_limit=25, offline=False, speak_no_speech=False):
    offline_mode = bool(offline)
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("Call prepare_workspace() before building the integration runtime.")
    enabled_text = os.environ.get("HEY_SWIFT_ENABLED_INTEGRATIONS")
    enabled = set(GROUPS) if enabled_text is None else set(filter(None, enabled_text.split(",")))
    if enabled - set(GROUPS):
        raise ValueError("Unknown enabled integration")
    speech = (SpeechService(PiperSpeech(Path(os.environ.get("VCM_VOICE_MODEL", str(ROOT / "assets/voices/en_US-amy-medium.onnx"))),
                                      phrases_dir=ROOT / "assets/speech"))
              if speak else None)
    if not live:
        return ActionRouter({group: PreviewService() for group in enabled}, speech=speech,
                            speak_no_speech=speak_no_speech)
    devices = (ScheduledDevices(PiHardware() if hardware == "pi" else MockHardware(), speech=speech)
               if "devices" in enabled else None)
    try:
        communications = communication_values()
        services = {}
        if devices is not None:
            services["devices"] = devices
        if "music" in enabled:
            selected = spotify_paths(ROOT)
            soloist = IntegrationSoloist(os.environ.get("VCM_SPOTIFY_URI", DEFAULT_SPOTIFY_URI),
                binary=str(selected["binary"]), data_dir=str(selected["data_dir"]),
                auto_start=True, key_file=selected["key_file"], profile=selected["profile"],
                max_volume_percent=music_volume_limit, operation_timeout=25.,
                startup_timeout=15., control_timeout=3., readiness_attempts=20)
            local = MpvBackend(sorted(Path(os.environ.get("VCM_MUSIC_DIR", str(ROOT / "assets/music"))).glob("*.mp3")),
                               str(ROOT / "tmp/mpv.sock"), auto_start=True, max_volume_percent=music_volume_limit)
            music = ManagedMusic(IntegrationAutoMusic(soloist, local,
                                 state_path=ROOT / "state/music-routing.json",
                                 reachable=(lambda: False) if offline_mode or os.environ.get("VCM_SPOTIFY_ENABLED") == "0" else (lambda: True)))
            try:
                music.backend.limit_active_volume()
            except Exception:
                pass
            services["music"] = music
        if "information" in enabled:
            weather = OpenMeteoWeatherProvider(offline=offline_mode or os.environ.get("VCM_WEATHER_ENABLED") == "0")
            services["information"] = InformationService(weather_provider=weather)
            weather.prefetch()
        if "reminders" in enabled:
            services["reminders"] = ReminderService(JsonReminderStore(ROOT / "state/reminders.json"))
        if "calling" in enabled:
            services["calling"] = CallingService(communications["call_number"])
        if "messaging" in enabled:
            services["messaging"] = ConfiguredMessaging(communications["phone_mac"], communications["message_number"])
        if os.environ.get("HEY_SWIFT_CONFIRM_COMMUNICATIONS") == "1":
            for name in ("calling", "messaging"):
                if name in services:
                    services[name] = ConfirmedCommunication(services[name])
        return ActionRouter(services, speech=speech, speak_no_speech=speak_no_speech)
    except Exception:
        if devices is not None:
            devices.close()
        raise
