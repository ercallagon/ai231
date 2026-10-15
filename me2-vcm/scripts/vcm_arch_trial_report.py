#!/usr/bin/env python
"""Generate the VCM architecture comparison report from experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path


RUNS = [
    ("run_006_e2_improved", "previous DS-CNN A1 champion"),
    ("run_014_e2_a1_full_aug", "intent-only A1 full-aug control"),
    ("run_016_e2_a2_wide", "intent-only wide TC-ResNet OOS control"),
    ("run_017_e2_a1_full_aug_slots", "slot-aware A1 full-aug control"),
    ("run_018_e2_a2_wide_slots", "slot-aware wide TC-ResNet control"),
    ("run_021_arch_a6_resnet2d_slots", "small 2D residual CNN"),
    ("run_022_arch_a7_mobilenetv2_slots", "MobileNetV2 tiny"),
    ("run_023_arch_a8_crnn_attn_slots", "attention CRNN"),
    ("run_024_arch_a9_conformer_tiny_slots", "tiny conformer/Transformer"),
    ("run_025_arch_a10_dsresnet_se_slots", "depthwise-separable ResNet + SE"),
    ("run_026_arch_a11_matchbox_slots", "MatchboxNet-lite"),
    ("run_027_arch_a6_resnet2d_slots_seed7", "small 2D residual CNN seed 7"),
    ("run_028_arch_a8_crnn_attn_slots_seed7", "attention CRNN seed 7"),
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def worst_classes(metrics: dict, n: int = 6) -> str:
    pc = metrics.get("per_class") or {}
    rows = sorted(
        ((name, vals.get("support", 0), vals.get("f1", 0.0)) for name, vals in pc.items()),
        key=lambda r: (r[2], -r[1], r[0]),
    )
    rows = [r for r in rows if r[1] > 0][:n]
    return ", ".join(f"{name} f1={f1:.3f} n={support}" for name, support, f1 in rows)


def top_confusions(metrics: dict, n: int = 10) -> str:
    conf = metrics.get("confusion") or {}
    pairs = []
    for k, v in conf.items():
        a, _, b = k.partition("->")
        if a != b:
            pairs.append((v, a, b))
    pairs.sort(reverse=True)
    return ", ".join(f"{a}->{b}:{v}" for v, a, b in pairs[:n])


def main() -> None:
    root = Path("experiments/vcm")
    lines = [
        "# VCM Architecture Trial Report",
        "",
        "Generated from repository artifacts. New architecture-screen runs are compared to the strongest completed controls.",
        "",
        "## Previous-Run Findings",
        "",
        "- DS-CNN A1 remains a strong intent-accuracy family, especially `run_014` on full augmentation.",
        "- Wide TC-ResNet (`run_016`) is the best completed OOS false-accept control, with far fewer OOS false positives than A1.",
        "- Slot-aware wide TC-ResNet (`run_018`) is the first completed model with strong slot accuracy, but OOS false accepts regress.",
        "- Compact old DS-CNN2D and the first CRNN/BC-ResNet attempts underperformed, so the new screen avoids rerunning those exact shapes.",
        "- Frozen slot-head training (`run_019`) showed that slots need a representation trained with slot pressure; attaching heads after freezing was insufficient.",
        "",
        "## Hypotheses Tested",
        "",
        "- `run_021`: a spatial residual CNN may improve local time-frequency discrimination without changing data.",
        "- `run_022`: a MobileNetV2-style model may keep enough accuracy with a better Pi 5 compute/size tradeoff.",
        "- `run_023`: attention pooling on a CRNN may help commands where the relevant slot token appears at different times.",
        "- `run_024`: a tiny conformer/Transformer-style encoder may improve longer-context command understanding.",
        "- `run_025`: depthwise-separable residual blocks plus squeeze-and-excitation may keep `run_021`'s local time-frequency strength while improving channel selectivity and deployment cost.",
        "- `run_026`: a MatchboxNet-style time-channel separable temporal CNN may approach CRNN robustness with simpler CPU/ONNX inference.",
        "- `run_027` and `run_028`: second-seed finalist checks test whether the best `run_021`/`run_023` findings are stable.",
        "",
        "## Literature Basis",
        "",
        "- BC-ResNet: broadcasted residual learning for efficient keyword spotting, used here as support for compact residual audio CNNs.",
        "- MobileNetV2: inverted residual and depthwise separable blocks for mobile inference.",
        "- Depthwise separable residual KWS with squeeze-and-excitation: direct support for the `run_025` hypothesis.",
        "- MatchboxNet: time-channel separable 1-D residual convolutions for speech command recognition.",
        "",
        "## Controlled Comparison",
        "",
        "| Run | Description | Arch | Params | Best val | Test acc | Macro-F1 | Slot acc | E2E acc | OOS P/R | OOS FP | CPU ms | ckpt MB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tag, desc in RUNS:
        d = root / tag
        res = load_json(d / "result.json")
        met = load_json(d / "metrics_test.json")
        lat = load_json(d / "latency_cpu.json")
        ckpt = d / "best.pt"
        lines.append(
            "| "
            + " | ".join([
                tag,
                desc,
                fmt(res.get("arch") or met.get("ckpt", {}).get("arch")),
                fmt(res.get("n_params") or met.get("ckpt", {}).get("n_params")),
                fmt(res.get("best_val_score")),
                fmt(met.get("intent_accuracy")),
                fmt(met.get("macro_f1")),
                fmt(met.get("slot_accuracy")),
                fmt(met.get("e2e_accuracy")),
                f"{fmt(met.get('oos_precision'))}/{fmt(met.get('oos_recall'))}" if met else "",
                fmt(met.get("oos_fp")),
                fmt(lat.get("mean_ms")),
                fmt(round(ckpt.stat().st_size / 1_000_000, 2)) if ckpt.exists() else "",
            ])
            + " |"
        )

    lines += [
        "",
        "## Per-Class And Error Analysis",
        "",
    ]
    for tag, desc in RUNS:
        met = load_json(root / tag / "metrics_test.json")
        if not met:
            continue
        lines.append(f"### {tag} - {desc}")
        lines.append("")
        lines.append(f"- Worst supported classes: {worst_classes(met)}")
        lines.append(f"- Top confusions: {top_confusions(met)}")
        lines.append("")

    lines += [
        "## Finalist Analysis",
        "",
        "Current completed finalists are `run_021` for best balanced offline score and `run_023` for a smaller/faster Pi-oriented tradeoff. Refinement runs test whether those conclusions survive SE/depthwise residual variants and second seeds.",
        "",
        "## Recommendation",
        "",
        "Current deployment reference is `run_021_arch_a6_resnet2d_slots`: it has the best completed intent accuracy and strong slot/e2e accuracy with much lower OOS false accepts than the slot-aware A1/TC-ResNet controls. `run_023_arch_a8_crnn_attn_slots` remains the compact fallback if Pi latency dominates.",
        "",
        "## Remaining Uncertainty",
        "",
        "- The screen is single-seed until finalists are identified.",
        "- CPU latency is a host proxy, not a Raspberry Pi 5 measurement.",
        "- OOS threshold calibration may still be needed after selecting the architecture.",
    ]
    out = Path("work/vcm_arch_trial/report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
