"""hey_swift — Hey Swift wake-word + Voice Command Model (VCM).

End-to-end, reproducible, open-source voice assistant pipeline for the
Raspberry Pi 5 (CPU-only):

    mic -> wake-word ("Hey Swift") -> VCM (19 intents + OOS + slots)
        -> action router -> typed device action -> feedback

Subpackages:
    wakeword   "Hey Swift" keyword-spotting subsystem
    vcm        voice command model (intent + closed-set slot + OOS)
    audio      shared capture / playback
    actions    intent + slot -> typed action routing
    hardware   Raspberry Pi peripherals (lazy-imported, optional)

Heavy/optional dependencies (torch, sounddevice, RPi.GPIO) are imported lazily
by the modules that need them, so `import hey_swift` works on a laptop with no
GPU and no Pi libs.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
