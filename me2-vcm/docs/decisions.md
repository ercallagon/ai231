# Decisions Log

Append-only record of the architectural and data decisions that define this
repo. Each entry states the decision, the rationale, and the evidence. New
entries are added at the bottom; entries are never rewritten (supersede with a
new entry that references the old one).

> This log was **reconstructed** on 2026-10-02 from the archived
> `me2-vcm2/config/base.yaml` and the `INTEGRATION_READINESS_REPORT.md` (there
> was no standalone decision file in the prior project). It is now the single
> source of truth for *why* the schema is what it is.

---

## D-001 — 19 intents + OUT_OF_SCOPE (20-way head)

**Decision:** The VCM classifies 19 in-scope intents plus one learned
`OUT_OF_SCOPE` class, for a 20-way softmax head.

**Rationale:** The 19 intents cover the full UX surface (music transport,
lights, climate, timers/alarms/reminders, info, comms). OOS is modeled as a
*learned class* rather than a post-hoc heuristic so the network can use acoustic
evidence to reject, not just a confidence threshold.

**Evidence:** All 19 intents + OOS are fully present in the HF dataset
(`airimonda/ai231-me2-voice-commands`); OOS has 270 train / 76 test / 15 holdout
dedicated rows plus 66,390 numeral rows (all OOS) as pressure/eval.

---

## D-002 — `slot_reject: false` (3-way slot heads) for the A0 baseline

**Decision:** For the A0 baseline, per-slot heads are **3-way** (supported
values only). The `REJECT` slot class is stripped from supervision.

**Rationale:** The HF dataset has only **158 empty-slot proxy rows** total,
unevenly distributed (ALARM 23, BRIGHTNESS 33, TIMER 37, TEMPERATURE 49,
CREATE_REMINDER 16, **COLOR 0**). A 4-way slot head would be starved of REJECT
examples — especially COLOR (0 rows) — and would learn a biased reject. With
`slot_reject: false`, rejection lives *entirely* in the OOS class + confidence
gate, which is well-supported.

**Escalation path (Tier-1 → Tier-2):** If the 3-way baseline shows unacceptable
slot misfires, enable `slot_reject: true` (4-way) **only after** synthesizing
REJECT rows (esp. COLOR) and fine-tuning the slot head *from the A0 backbone*
— not a from-scratch retrain. This is a controlled follow-up experiment, not the
default.

**Effect on data:** The 158 empty-slot proxy rows are dropped from *slot*
supervision but kept as intent-only rows.

---

## D-003 — OOS = learned class + confidence gate (τ = 0.5)

**Decision:** An utterance is **accepted** iff `argmax ∈ 0..18` **and**
`softmax_max ≥ τ`, with τ initialized to 0.5, tuned on the frozen validation
split and verified on holdout.

**Rationale:** Two independent rejection signals (learned OOS class + confidence
floor) are more robust than either alone. The gate catches low-confidence
in-scope guesses that the OOS class misses.

**Note:** Intent-level OOS and slot-level validity are independent — a command
can be a valid intent but still have an invalid/missing slot (handled by the
action router's slot validation, see D-005).

---

## D-004 — 2.0 s input window (not 1.5 s)

**Decision:** The VCM input window is **2.0 s** (32,000 samples → 197 frames →
`(1, 80, 197)`), with 1.5 s retained as a control.

**Rationale:** The HF duration distribution argues against 1.5 s — many slotted
commands ("set the temperature to 22 degrees") exceed 1.5 s and would be
truncated. 2.0 s covers the bulk of the distribution with center-crop/pad to a
fixed width.

---

## D-005 — Closed-set slots, validated at the router

**Decision:** Slots are closed-set (3 supported values per slotted intent). The
action router validates a predicted slot against the intent's declared set and
rejects invalid values with a structured result (not an exception).

**Rationale:** Closed-set slots keep the slot heads small and the deployment
deterministic. Open-ended slots (free-text reminder) are a Phase-2 addition via
a secondary ASR path, not the default.

---

## D-006 — A0 = DS-CNN (32,64,128,256), ~57k params, from scratch

**Decision:** The first run (`a0_scratch`) is a dilated-separable CNN with
channel widths (32, 64, 128, 256), ~57k params, trained **from scratch** (no
weight transfer), seed 42, 30 epochs, batch 64, lr 1e-3, Adam, early-stop
patience 6.

**Rationale:** Compute is cheap (prior runs took minutes), so we run a small
pre-registered matrix where each run answers one question. A0 is the baseline:
"is the current capacity enough for 19 intents?" A1 (wider, ~123k) and A2
(TC-ResNet w32, ~131k) are follow-ups.

**Hard rule:** Phase 1 is from-scratch. Weight transfer from the old s3v3/s3v4
finalists is a *later controlled experiment*, never the default.

---

## D-007 — HF-only data; frozen test + holdout never trained on

**Decision:** Training uses **only** the HF dataset. The local OptionB corpus is
excluded entirely. The frozen test split (4,443 rows) and untouched holdout
(202 rows) are **eval-only** and never enter training.

**Rationale:** Segregation keeps the experiment clean and reproducible. The old
local-corpus projects are preserved read-only in the archive for a *post-hoc*
baseline comparison only (different dataset → not directly comparable).

---

## D-008 — Immutable, tagged experiment artifacts

**Decision:** Every run writes to `experiments/<target>/<tag>/` and is written
**once**; a re-run uses a new tag. `reproducibility.json` records seed, config,
HF revision, split counts, and the disjointness proof.

**Rationale:** Immutable artifacts make any historical result auditable and
prevent silent overwrite. The disjointness proof (train/val/test/holdout
audio-disjoint) is recorded at train time, not assumed.

---

## D-009 — Direct audio classification (not ASR→NLU)

**Decision:** Intent is classified **directly from audio** (end-to-end), not via
an ASR transcription followed by NLU.

**Rationale (from the integration report §5.2):** A direct-audio intent
classifier is simpler, faster to converge, and fully on-device (no ASR
dependency on the Pi). ASR is reserved as a *secondary* path for open-ended slot
values (Phase 2). References: FSC (Lugosch 2019) motivates intent+slot; Speech
Commands (Warden 2018) validates the unknown-class + reproducible-eval approach.

---

## D-010 — Wake-word: NumPy inference, sustain/cooldown/RMS gating

**Decision:** The "Hey Swift" wake-word detector runs **NumPy-only** at
inference (Pi-clean, no torch). Runtime gating: RMS floor −60 dB, p(wake) ≥
0.30, sustain 2 consecutive hops, cooldown 2.0 s.

**Rationale (carry-forward from wakeword_exp15_mined_lp, ~204,738 params):**
These gates were validated to eliminate false triggers and double-fires. The
model is a Snips-style dilated CNN (Hey Snips, Coucke 2019) with a
(1, 80, 100) input (40-mel × 2 delta × 100 frames, 1.0 s @ 16 kHz).

---

## D-011 — Lazy hardware imports (import-safe everywhere)

**Decision:** `hardware/` (RPi.GPIO, spidev, smbus) and `audio/` (sounddevice)
are imported **lazily**, inside methods, never at module import. `import
hey_swift` must succeed on a laptop with no GPU and no Pi libs.

**Rationale:** Open-source usability + testability. The test suite and any
developer machine run headless; a clear error is raised only when a
Pi-specific or audio action is actually invoked.

---

## D-012 — SHA-pinned dataset fetch; only manifests committed

**Decision:** `scripts/01_download_hf.py` pins
`airimonda/ai231-me2-voice-commands` @ revision `bfca0bbe…` and verifies each
file's size + LFS SHA-256 oid. Only `data/**/manifests/` (small CSVs + `splits.json`)
is committed; the heavy parquet is gitignored and re-fetchable.

**Rationale:** Reproducibility. A re-run months later pulls byte-identical data,
and the committed manifests pin the splits. Bumping the HF revision requires a
new experiment tag (see D-008).
