"""Read-only readiness probes; successful imports do not verify hardware effects."""
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import shutil
import subprocess


def diagnose(root, home, cfg=None):
    from .deployment import isolated_env, applied_env, microphone, private_json
    checks = []

    def check(name, probe, scope, pending=False):
        try:
            value = probe()
            checks.append(dict(name=name, status="PENDING" if pending else "PASS", scope=scope, detail=value))
        except Exception:
            checks.append(dict(name=name, status="MISSING", scope=scope,
                               detail="Probe failed; see quickstart prerequisites. No private configuration included."))

    for name in ("numpy", "scipy", "onnxruntime", "soundfile", "sounddevice", "pyarrow", "yaml", "gpiozero", "dbus_next"):
        check("import:" + name, lambda n=name: bool(importlib.import_module(n)), "actual import")
    check("lgpio", lambda: importlib.import_module("lgpio").__file__, "OS-provided lgpio ABI import; not GPIO actuation")
    commands = ["mpv", "wpctl"]
    if cfg and set(cfg["enabled"]) & {"calling", "messaging"}:
        commands += ["bluetoothctl", "busctl"]
    for name in commands:
        checks.append(dict(name="executable:" + name, status="PASS" if shutil.which(name) else "MISSING",
                           scope="available executable", detail="No system state changed"))
    registry = json.loads((root / "configs/assistant_profiles.json").read_text())["profiles"]
    for name, paths in registry.items():
        def contract(p=paths):
            from .vcm.inference import VCMInference
            import numpy as np
            model = VCMInference(str(root / p["config"]), str(root / p["model"]), str(root / p["labels"]))
            result = model.predict((.1*np.sin(np.arange(16000)*2*np.pi*440/16000)).astype(np.float32))
            if not np.isfinite(result["confidence"]):
                raise ValueError("Nonfinite model output")
            return dict(threshold=model.tau, inputs=[dict(name=x.name, shape=x.shape) for x in model.sess.get_inputs()])
        check("model:" + name, contract, "ONNX/config/label load and finite-input execution; no accuracy claim")
    def wake():
        import numpy as np
        from .wakeword.inference import WakeWordDetector
        detector = WakeWordDetector(str(root / "configs/wakeword.yaml"), onnx_path=str(root / "models/wakeword/wakeword_exp15_mined_lp.onnx"))
        return bool(np.isfinite(detector._score_window(np.zeros(detector.win, np.float32))[:2]).all())
    check("wake:exp15", wake, "ONNX contract and synthetic scoring; not wake recall")
    if cfg:
        def mic():
            import sounddevice as sd
            index = int(microphone(cfg))
            rates = []
            for rate in (16000, 48000):
                try:
                    sd.check_input_settings(device=index, channels=1, samplerate=rate, dtype="float32")
                    rates.append(rate)
                except Exception:
                    pass
            if not rates:
                raise ValueError("No supported input rate")
            return dict(index=index, rates=rates)
        check("microphone", mic, "stable device identity and input format support; audible capture pending")
        with applied_env(isolated_env(cfg, home / "runtime")):
            if cfg.get("weather"):
                def weather():
                    from .actions.services.weather import OpenMeteoWeatherProvider
                    OpenMeteoWeatherProvider().current_weather()
                    return "Current weather fetched; no key required"
                check("weather", weather, "read-only network query")
            if cfg.get("speech"):
                def voice():
                    from piper import PiperVoice
                    PiperVoice.load(cfg["voice_model"], use_cuda=False)
                    return "Voice loads; audible playback pending"
                check("Piper", voice, "voice/config load, no playback")
            if cfg.get("spotify"):
                def spotify():
                    from .actions.services.music import SoloistBackend
                    backend = SoloistBackend(binary=cfg["soloist_binary"], data_dir=str(home / "runtime/state/soloist"))
                    if not backend.is_running():
                        raise ValueError("Run swift spotify")
                    state = backend.ctl("now", as_json=True, timeout=2)
                    if not state.get("is_active"):
                        raise ValueError("Configured Spotify device is not active")
                    return {k: state.get(k) for k in ("status", "is_active", "volume")}
                check("Spotify", spotify, "existing private daemon status; does not activate or play")
            if "calling" in cfg["enabled"]:
                def calls():
                    from .actions.services.calling import BluetoothCallingBackend
                    data = BluetoothCallingBackend(timeout=3)._call("/org/pipewire/Telephony", "org.ofono.Manager", "GetModems")
                    modems = data.get("data", [{}])[0]
                    if not isinstance(modems, dict) or len(modems) != 1:
                        raise ValueError("One compatible gateway required")
                    return "One gateway responds; no Dial sent"
                check("calling", calls, "read-only telephony availability; real call pending explicit approval")
            if "messaging" in cfg["enabled"]:
                def message():
                    from .actions.services.messaging import BluetoothMessagingBackend
                    BluetoothMessagingBackend(cfg["secrets"]["phone_mac"], timeout=4, total_timeout=6).check()
                    return "MAP interface responded; no message sent"
                check("messaging", message, "MAP availability; phone permission may be needed; no PushMessage")
        if cfg["hardware"] == "pi":
            check("GPIO/display", lambda: "BCM17/27/22 and BCM23/24 preserved; physical effects not exercised", "physical wiring/effects", pending=True)
    else:
        checks.append(dict(name="first-run setup", status="MISSING", scope="configuration", detail="Run swift.py setup"))
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), checks=checks,
                  setup_mode="separate checkout/venv/state on shared OS; not fresh-OS verification",
                  physical_checks="PENDING unless separately recorded; no calls/SMS placed")
    destination = root / "work/pi5_test_01/DIAGNOSTICS.json"
    private_json(destination, result)
    for item in checks:
        print(f"{item['status']:7} {item['name']}: {item['scope']}")
    print("Diagnostic report:", destination)
    return int(any(c["status"] == "MISSING" for c in checks))
