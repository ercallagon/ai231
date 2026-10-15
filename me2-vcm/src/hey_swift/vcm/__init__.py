"""hey_swift.vcm — Voice Command Model subsystem.

19-intent classification + per-intent closed-set slot extraction + explicit
OUT_OF_SCOPE rejection, trained on the HF dataset only.

Modules:
    taxonomy      intents, OOS, and slot definitions (single source of truth)
    features      audio -> mel spectrogram features
    dataset       HF dataset loading + split handling
    augmentation  training-only audio augmentation
    models        model architectures (DS-CNN, TC-ResNet)
    train         reproducible training loop
    evaluate      intent / slot / OOS evaluation
    inference     live / pretrained inference + ONNX export
"""
