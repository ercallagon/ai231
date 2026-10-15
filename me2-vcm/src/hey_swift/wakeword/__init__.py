"""hey_swift.wakeword — "Hey Swift" keyword-spotting subsystem.

Detects the "Hey Swift" wake phrase from a live microphone stream and signals
the VCM to listen for a command. The deployed detector is NumPy-only (Pi-clean);
training is PyTorch (dilated CNN, Snips-style).

Modules:
    dataset      wake-word dataset loading
    features     audio -> 40-mel + 2-delta features (NumPy)
    model        dilated CNN architecture
    train        reproducible training loop
    evaluate     accuracy / FAR / FRR evaluation
    inference    live / pretrained detection (sustain + cooldown + RMS gate)
"""
