# Swift ME3 — Laptop Browser AEC Qualification (Test B)

**Date:** 2026-11-13 (Asia/Manila)  
**Result:** **PASS — laptop/browser AEC for the operator-tested USB microphone and speaker configuration**, with scope limits below.  
**Evidence source:** Operator-run tests, terminal outputs, browser JSON metrics and listening feedback supplied in the conversation. This document does **not** claim independent inspection of the original WAV/WebM or JSON bytes.

## Scope and decision

The browser-local `/audio-probe` test measured whether browser echo cancellation reduced audible speaker-to-microphone feedback while preserving the operator's speech during overlap. Both the original boiling-speech reference and a generated **Deepgram Aura-2 Thalia** speech reference were tested with AEC off and on. The operator confirmed for the **Thalia AEC-On** recording that the reference voice was noticeably quieter, their own voice remained **clear and intelligible during double-talk (8–16 s)**, and their own voice remained **clear after playback stopped (16–24 s)**.

**Qualification:** PASS for the **laptop browser AEC diagnostic configuration tested**. This does **not** qualify native Pipecat end-to-end conversation, real-time live-ASR/TTS latency, native barge-in or interruption, Raspberry Pi 5 AEC, physical speaker loudness, or performance on other devices/rooms.

## Test setup

- Existing Windows laptop USB microphone and USB speaker; browser's AEC Off/On toggle (no headphones).
- The standalone, loopback-only ME3 dashboard served `/audio-probe` at `http://127.0.0.1:8766/audio-probe`, independent of the native Pipecat bridge on port 8765.
- The native bridge on 8765 did **not** serve `/audio-probe` (404) because `build_app` delegates native mode to `build_native_app`, which does not register the route; the standalone `dashboard.py serve --port 8766` handler does.
- Browser JSON reported `requestedAEC = reportedAEC = False` for Off recordings, `True` for On recordings; `noiseSuppression = False`, `autoGainControl = False` for all four.
- Probe phases: 0–8 s operator silent / reference playback; 8–16 s simultaneous operator speech and reference playback; 16–24 s operator speech after reference playback stopped.
- Exact browser build, USB device identifiers, speaker SPL/volume, and room conditions were not captured in the supplied outputs; these are limits of generalizability.

**Follow-up:** On 2026-11-13 the native bridge registered the same existing probe renderer at `/audio-probe`, retaining its local-request validation, CSP, microphone Permissions-Policy and no-upload page behavior. An offline HTTP regression covers the route and headers. This corrects the earlier 8765 route gap; it does not alter or extend the operator's acoustic PASS evidence.

## Evidence locations (HPC repo-relative)

Root: `var/results/aec/laptop_20261113/`

| ID | Folder | WebM recording | JSON diagnostics | Bytes (WebM) | Packet-derived duration |
|---|---|---|---|---:|---:|
| 1 | `laptop-aec-test1-boiling/` | `laptop-aec-off-1.webm` | `laptop-aec-off-1.json` | 386,545 | 23.94 s |
| 2 | `laptop-aec-test1-boiling/` | `laptop-aec-on-2.webm` | `laptop-aec-on-2.json` | 386,545 | 23.94 s |
| 3 | `laptop-aec-test2-thalia/` | `laptop-aec-off-3.webm` | `laptop-aec-off-3.json` | 387,511 | 24.00 s |
| 4 | `laptop-aec-test2-thalia/` | `laptop-aec-on-4.webm` | `laptop-aec-on-4.json` | 386,545 | 23.94 s |

All four WebM files decoded successfully with FFmpeg. Container `format.duration` reported `N/A`, which is permissible for browser MediaRecorder WebM; packet timestamps yielded the durations above. All SHA-256 digests were distinct:

```text
e61bf84fc4e13e629a1592db4227adf34a397e0585a796d513258703972df4ab  laptop-aec-off-1.webm
097f495a817f387f2a8f47ab322d1c1eaf7cedee6e434dd8efdb44125d55f5ea  laptop-aec-on-2.webm
936aaed2fc4a69aa11e3c2c9ee9ff28447b52473c9e73a6386e528f7186952a9  laptop-aec-off-3.webm
d2ab896e76258daf5bfe62b65a426494b475f8bddb19a54da4a959e2264c72db  laptop-aec-on-4.webm
```

Reference source: first trial used the earlier guided 'boiling' human speech recording, which had substantial quiet gaps. Second used reusable **Aura-2 Thalia** speech generated for this test. The latter is at:

```text
var/results/aec/swift_aec_aura2_thalia.wav
var/results/aec/swift_aec_aura2_thalia.webm
```

Both reference files were manually played and found approximately **35 s** in duration. One Deepgram TTS generation request was made to produce the Thalia fixture. The local AEC tests did not call speech/LLM providers.

## Measurements

All figures are 8-second-window decoded microphone **RMS levels in dBFS**, not a formal ERLE measurement or calibrated acoustic response.

| Trial / AEC | Far-end only 0–8 s | Double-talk 8–16 s | Near-end only 16–24 s |
|---|---:|---:|---:|
| Boiling / Off | −45.1 | −22.8 | −26.1 |
| Boiling / On | −63.7 | −24.2 | −27.8 |
| **Boiling / On − Off** | **−18.6 dB** | −1.4 dB | −1.7 dB |
| Thalia / Off | −42.6 | −23.6 | −24.6 |
| Thalia / On | −65.1 | −22.3 | −25.4 |
| **Thalia / On − Off** | **−22.5 dB** | +1.3 dB | −0.8 dB |

- The **speaker-only** interval recorded 18.6 dB lower RMS for Boiling, and 22.5 dB lower RMS for Thalia, when browser AEC was on. The latter difference is about 13.3× lower RMS amplitude. These are *recorded microphone-level differences*, not calibrated ERLE.
- During Thalia double-talk the total recorded RMS changed +1.3 dB, while near-end-only changed −0.8 dB; these levels alone cannot establish intelligibility or separation of near/far speech.
- Operator listening of the **Thalia** files supplied the key additional evidence: clear and intelligible near-end speech with AEC On during double-talk, and clear speech after reference playback stopped. No equivalent explicit listening ratings were provided for Boiling.

## Assessment, limitations, follow-ups

**PASS:** Browser AEC toggling, diagnostic capture integrity, strong far-end microphone-level attenuation, and Thalia-trial operator-confirmed double-talk and near-end intelligibility under this specific laptop USB setup.

**NOT YET PASS/NOT TESTED:** Native Pipecat AEC or interruption, live-provider end-to-end task and latency, reliable ASR barge-in, Raspberry Pi 5 playback/reference routing, physical speaker AEC or delay, cross-machine repeatability. Avoid saying these were qualified by this Test B.

**Not established:** Calibrated echo return loss enhancement (ERLE), time-aligned reference and mic cross-correlation, objective speech intelligibility scores, or identical acoustic excitation between independent Off/On runs. Microphone RMS includes near-end speech and environmental noise. End-user listening remains subjective evidence.

**Engineering observations to retain:**
1. **Native bridge missing route:** `/audio-probe` HTTP 404 at port 8765 native mode; standalone dashboard at 8766 worked. Correct native route exposure only if justified and covered by offline route regression tests. No second long-lived service is desired for normal deployment.
2. **Thalia generator WAV duration anomaly:** A valid, approximately 35 s generated WAV/WebM was printed as `WAV: 44737.9s, 24000 Hz, 1 channel(s)` by `scripts/create_swift_thalia_aec_reference.py`. Suspect streaming WAV header placeholder or header-based duration accounting; investigate rather than assuming exact cause. Inspect any production code parsing WAV metadata for audio duration, pacing, interruption, timeouts, metrics, and playback. Do not regenerate the reference merely to reproduce this issue.
3. **Preserve reusable fixture:** Treat reference as a private golden audio fixture for deterministic offline AEC, playback and barge-in tests, with metadata; do not commit operator-recorded voice to a public repository.

**Recommendation:** Preserve original WebM and JSON files unchanged, attach or archive this report privately with them, and move to a separate native bridge/live-audio qualification only after the known ASR timeout discrepancy is resolved and a distinct provider-call budget is explicitly approved.
