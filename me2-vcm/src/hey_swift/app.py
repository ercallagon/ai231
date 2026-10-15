"""End-to-end Hey Swift application: wake -> command -> action -> feedback.

This is the live runtime glue. It wires the wake-word detector, the VCM
inference, the action router, and (optionally) the hardware feedback loop.

On a laptop without a microphone or Pi libs, `main()` raises a clear, actionable
error rather than failing at import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class PipelineState:
    """Mutable state threaded through the live loop."""
    armed: bool = False
    last_fire_ts: float = 0.0
    pending_command: Optional[str] = None
    history: list = field(default_factory=list)


def build_pipeline(config_paths: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Construct the runtime components (lazy imports keep this import-safe).

    Returns a dict of wired components: wake, vcm, router, feedback.
    """
    from hey_swift.wakeword.inference import WakeWordDetector
    from hey_swift.vcm.inference import VCMInference
    from hey_swift.actions.router import ActionRouter

    cfg = config_paths or {}
    wake = WakeWordDetector(cfg.get("wakeword", "configs/wakeword.yaml"))
    vcm = VCMInference(cfg.get("vcm", "configs/vcm.yaml"))
    router = ActionRouter(cfg.get("actions", "configs/actions.yaml"))
    return {"wake": wake, "vcm": vcm, "router": router}


def handle_wake(state: PipelineState, ts: float, cooldown: float = 2.0) -> bool:
    """Apply the sustain/cooldown gate. Returns True if the wake should act."""
    if ts - state.last_fire_ts < cooldown:
        return False
    state.last_fire_ts = ts
    state.armed = True
    return True


def dispatch(vcm_prediction: Dict[str, Any], router: Any) -> Dict[str, Any]:
    """Route a VCM prediction (intent + slots + confidence) to an action."""
    return router.route(vcm_prediction)


def main() -> None:
    """Run the live loop. Requires a microphone (hey-swift[audio])."""
    try:
        from hey_swift.audio.capture import CommandCapture
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Live capture needs the 'audio' extra. Install with:\n"
            "    pip install 'hey-swift[audio]'\n"
            f"(underlying error: {exc})"
        )

    comps = build_pipeline()
    state = PipelineState()
    capture = CommandCapture("configs/hardware.yaml")

    # Speaker feedback (real beep on the Pi; silent no-op fallback headless).
    try:
        from hey_swift.audio.playback import make_playback
        playback = make_playback("configs/hardware.yaml")
    except Exception:
        playback = None

    console_print = print
    console_print("[hey-swift] live loop starting; say 'Hey Swift' then a command.")

    for audio_chunk in capture.stream():
        if not state.armed:
            # NOTE: the detector exposes process() -> (events, p_last); there is
            # no detect() method. A fire is signalled by a non-empty event list.
            events, _p_last = comps["wake"].process(audio_chunk)
            if events and handle_wake(state, capture.timestamp()):
                if playback is not None:
                    try:
                        playback.beep()
                    except Exception:
                        pass
                console_print("[hey-swift] wake detected — listening for command…")
        else:
            command_audio = capture.capture_command()
            prediction = comps["vcm"].predict(command_audio)
            result = dispatch(prediction, comps["router"])
            console_print(f"[hey-swift] {prediction['intent']} -> {result}")
            state.armed = False
            state.history.append((prediction, result))
