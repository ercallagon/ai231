"""VCM inference + ONNX export.

Two runtime paths:
    * torch path  — used by evaluate.py and as the export source.
    * ONNX path   — the Pi deployment path (CPU-only, no torch).

`export_onnx(tag)` traces the torch model to ONNX and writes it to
models/vcm/<tag>.onnx plus a labels.json (intent order + slot sets + tau) so the
deploy bundle is self-describing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import yaml

from .features import FeatureConfig, MelFeatures
from .taxonomy import load_taxonomy

MODELS_DIR = Path("models/vcm")
EXP_ROOT = Path("experiments/vcm")


class VCMInference:
    """ONNX-based VCM inference for the Pi (CPU-only)."""

    def __init__(self, config: str = "configs/vcm.yaml",
                 onnx_path: Optional[str] = None,
                 labels_path: Optional[str] = None):
        import onnxruntime as ort

        self.config = config
        self.cfg = yaml.safe_load(open(config, encoding="utf-8")) or {}
        self.tax = load_taxonomy(config)
        self.feat_cfg = FeatureConfig.from_dict(self.cfg.get("audio", {}))
        self.frontend = self.cfg.get("runtime", {}).get("frontend", "legacy_center_crop")
        if self.frontend == "whole6s_mask600_final_partial_v1":
            from .fullcontext import FullContextFeatures
            if (self.feat_cfg.window_seconds != 6.0 or self.feat_cfg.sample_rate != 16000
                    or self.feat_cfg.n_fft != 400 or self.feat_cfg.hop_length != 160
                    or self.feat_cfg.n_mels != 80 or self.feat_cfg.f_min != 20
                    or self.feat_cfg.f_max != 8000):
                raise ValueError("Full-context model requires its frozen six-second frontend")
            self.feats = FullContextFeatures()
        elif self.frontend == "legacy_center_crop":
            self.feats = MelFeatures(self.feat_cfg)
        else:
            raise ValueError("Unknown model frontend")
        self.tau = float(self.cfg.get("oos", {}).get("confidence_threshold", 0.5))

        path = onnx_path or str(MODELS_DIR / "selected.onnx")
        labels = Path(labels_path) if labels_path else Path(path).with_suffix(".labels.json")
        if not labels.exists():
            alt = Path(path).with_name(Path(path).stem + ".labels.json")
            labels = alt if alt.exists() else labels
        if not labels.exists():
            raise FileNotFoundError("Required model labels are missing: " + str(labels))
        self.labels = json.loads(labels.read_text())
        self.intent_labels = list(self.labels.get("intents", self.tax.intents))
        if self.tax.oos_class not in self.intent_labels:
            self.intent_labels.append(self.tax.oos_class)
        if self.intent_labels != list(self.tax.intents) + [self.tax.oos_class]:
            raise ValueError("Model/config intent order mismatch")
        self.slot_outputs = self.labels.get("slot_outputs", {n: "slot_" + n for n in self.tax.slotted_intents})
        self.slot_sets = self.labels.get("slot_sets", {k: self.tax.slot_classes(k) for k in self.tax.slotted_intents})
        if "confidence_threshold" in self.labels and abs(float(self.labels["confidence_threshold"])-self.tau) > 1e-9:
            raise ValueError("Config/labels rejection threshold mismatch")
        options = ort.SessionOptions()
        self.threads = int(self.cfg.get("runtime", {}).get("threads", 1))
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sess = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]
        expected_inputs = {"mel", "frame_mask"} if self.frontend != "legacy_center_crop" else {"mel"}
        if {i.name for i in self.sess.get_inputs()} != expected_inputs:
            raise ValueError("ONNX/frontend input contract mismatch")
        expected_frames = 600 if self.frontend != "legacy_center_crop" else 1 + (self.feat_cfg.window_samples-self.feat_cfg.n_fft)//self.feat_cfg.hop_length
        if list(self.sess.get_inputs()[0].shape[1:]) != [1, self.feat_cfg.n_mels, expected_frames]:
            raise ValueError("ONNX/frontend feature shape mismatch")
        output_shapes = {o.name:o.shape for o in self.sess.get_outputs()}
        if output_shapes.get("intent_logits", [0,0])[-1] != self.tax.n_way_head:
            raise ValueError("ONNX intent output width mismatch")
        for name, out_name in self.slot_outputs.items():
            if output_shapes.get(out_name, [0,0])[-1] != len(self.slot_sets[name]):
                raise ValueError("ONNX slot width mismatch: " + name)

    def predict(self, wav) -> Dict:
        """Return {intent, confidence, accepted, slots} for a waveform."""
        import numpy as np

        wave = np.asarray(wav, dtype="float32")
        if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all():
            return self._reject("invalid_audio")
        if self.frontend != "legacy_center_crop":
            mel, mask, receipt = self.feats.transform(wave)
            if mel is None:
                return self._reject(receipt["status"])
            feed = {"mel": mel[None], "frame_mask": mask[None]}
        else:
            mel = self.feats.transform(wave)
            feed = {self.input_name: mel[None]}
        if mel.ndim == 3:
            mel = mel[None, :, :, :]
        outs = self.sess.run(None, feed)
        if not all(np.isfinite(value).all() for value in outs):
            return self._reject("nonfinite_model_output")
        by_name = dict(zip(self.output_names, outs))
        logits = by_name.get("intent_logits", outs[0])[0]
        probs = _softmax(logits)
        conf, idx = float(probs.max()), int(probs.argmax())
        accepted = (idx != self.tax.oos_idx) and (conf >= self.tau)
        intent = self.intent_labels[idx] if idx < len(self.intent_labels) else self.tax.oos_class
        slots = {}
        slot_conf = None
        if accepted and intent in self.slot_outputs:
            out_name = self.slot_outputs[intent]
            slot_logits = by_name[out_name][0]
            slot_probs = _softmax(slot_logits)
            sid = int(slot_probs.argmax())
            vals = self.slot_sets.get(intent, [])
            if sid < len(vals):
                slots[intent] = vals[sid]
                slot_conf = float(slot_probs[sid])
            if sid >= len(vals) or vals[sid] == "REJECT":
                accepted = False
                slots = {}
                return {"intent": intent, "confidence": conf, "accepted": False, "oos": False,
                        "slots": {}, "slot_confidence": slot_conf, "rejection_reason": "unsupported_slot"}
        return {
            "intent": intent,
            "confidence": conf,
            "accepted": bool(accepted),
            "oos": (idx == self.tax.oos_idx),
            "slots": slots,
            "slot_confidence": round(slot_conf, 4) if slot_conf is not None else None,
        }

    @staticmethod
    def _reject(reason):
        return {"intent": "OUT_OF_SCOPE", "confidence": 0., "accepted": False,
                "oos": True, "slots": {}, "slot_confidence": None, "rejection_reason": reason}


def export_onnx(tag: str = "a0_scratch", config: str = "configs/vcm.yaml") -> Path:
    """Trace the trained torch model to ONNX and write labels.json."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("ONNX export requires torch: pip install 'hey-swift[train]'") from exc

    from .models import ArchSpec, build_model

    tax = load_taxonomy(config)
    ckpt = EXP_ROOT / tag / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint found at {ckpt}")
    payload = torch.load(ckpt, map_location="cpu")
    arch_name = payload.get("arch", "ds_cnn")
    n_params = int(payload.get("n_params", 0) or 0)
    n_way = int(payload.get("n_way", tax.n_way_head))
    if arch_name == "ds_cnn":
        spec = ArchSpec.a1(n_way=n_way) if n_params > 800_000 else ArchSpec.a0(n_way=n_way)
    elif arch_name == "tc_resnet":
        spec = ArchSpec("tc_resnet", [64, 64, 64, 64], width=64, n_way_head=n_way) if n_params > 150_000 else ArchSpec.a2(n_way=n_way)
    elif arch_name == "crnn":
        spec = ArchSpec.a3(n_way=n_way)
    elif arch_name == "bc_resnet":
        spec = ArchSpec.a4(n_way=n_way)
    elif arch_name == "ds_cnn_sep":
        spec = ArchSpec.a5(n_way=n_way)
    elif arch_name == "resnet2d_small":
        spec = ArchSpec.a6(n_way=n_way)
    elif arch_name == "mobilenetv2_tiny":
        spec = ArchSpec.a7(n_way=n_way)
    elif arch_name == "crnn_attn":
        spec = ArchSpec.a8(n_way=n_way)
    elif arch_name == "conformer_tiny":
        spec = ArchSpec.a9(n_way=n_way)
    elif arch_name == "ds_resnet_se":
        spec = ArchSpec.a10(n_way=n_way)
    elif arch_name == "matchboxnet":
        spec = ArchSpec.a11(n_way=n_way)
    else:
        spec = ArchSpec.a0(n_way=n_way)
    slot_heads = payload.get("slot_heads") or {}
    slot_head_labels = {intent: tax.slot_classes(intent) for intent in slot_heads if tax.slot_classes(intent)}
    model = build_model(spec, slot_heads=slot_head_labels or None)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    class ExportWrapper(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped
            self.slot_names = list(getattr(wrapped, "slot_head_map", {}).keys())

        def forward(self, x):
            out = self.wrapped(x)
            if not isinstance(out, dict):
                return out
            return tuple([out["intent"]] + [out["slots"][name] for name in self.slot_names])

    wrapper = ExportWrapper(model)

    out_dir = MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{tag}.onnx"

    feat_cfg = FeatureConfig.from_dict(yaml.safe_load(open(config, encoding="utf-8")).get("audio", {}))
    frames = int(feat_cfg.window_seconds * feat_cfg.sample_rate // feat_cfg.hop_length) - 2
    dummy = torch.randn(1, 1, feat_cfg.n_mels, frames)
    output_names = ["intent_logits"] + [f"slot_{name}" for name in wrapper.slot_names]
    dynamic_axes = {"mel": {0: "batch"}, "intent_logits": {0: "batch"}}
    for name in output_names[1:]:
        dynamic_axes[name] = {0: "batch"}
    torch.onnx.export(
        wrapper, dummy, str(onnx_path),
        input_names=["mel"], output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
    )

    (out_dir / f"{tag}.labels.json").write_text(json.dumps({
        "intents": list(tax.intents) + [tax.oos_class],
        "oos_class": tax.oos_class,
        "n_way_head": tax.n_way_head,
        "slot_reject": tax.slot_reject,
        "slot_sets": {k: tax.slot_classes(k) for k in tax.slotted_intents if tax.slot_classes(k)},
        "slot_outputs": {name: f"slot_{name}" for name in wrapper.slot_names},
        "confidence_threshold": float(
            yaml.safe_load(open(config, encoding="utf-8")).get("oos", {}).get("confidence_threshold", 0.5)
        ),
        "input_shape": [1, 1, feat_cfg.n_mels, frames],
        "arch": arch_name,
        "n_params": n_params,
    }, indent=2))

    print(f"[export] {onnx_path} (+ {tag}.labels.json)")
    return onnx_path


def _softmax(x):
    import numpy as np
    e = np.exp(x - x.max())
    return e / e.sum()
