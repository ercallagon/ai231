# Existing audio and remaining capture needs

Inventory date: 2026-11-09. Scope: this ME3 checkout's `var/recordings`, `var/experiments`, and `var/results`; no sibling project, cache or remote Pi filesystem. Method: file hashes, WAV headers, ffprobe metadata/packet timestamps, guided manifests, and existing experiment reports. **No new listening, inference, provider upload or physical capture.** Full private manifest: `var/results/architecture_modernization/phase2-audio-inventory-complete.json`. Reproduce with `python -B scripts/phase2/audio_inventory.py --output var/results/audio-inventory-FRESH.json` using the existing environment. ffprobe is optional; no package is installed. Packet timestamp extent is not a decoded-sample measurement.

## What already exists

| Material | Verified inventory | Reuse and limits |
|---|---|---|
| Guided browser human speech | **54 takes, 435.923 s (7 min16 s)** across six manifests: core6, timing7, actions14, tasks9, agent5, supplemental13. All54 present, complete flags true, hashes match. Mono48k Opus; all reported browser AEC/NS/AGC enabled. | Strong starting corpus for ASR comparisons, intent/context regression, natural pauses, follow-ups, corrections, numbers, music/device/task controls and speech-stop versus task-cancel phrases. Expected prompts exist, but are not independently listened/word-aligned ground truth. Processing cannot be undone; not raw Pi capture. |
| Quiet browser reading | `swift-asr-reading-1.webm`,18.0s, mono48k Opus; matching JSON supplies26-word reference and reports AEC/NS/AGC off. | Existing longer speech/wake/ASR fixture; manual reference confirmation still needed. Historical university response in experiment006 is one provider result, not a fresh comparison. |
| Laptop AEC pair | `laptop-aec-off-1.webm` and `laptop-aec-on-2.webm`, approximately23.94s each; settings sidecars report AEC off/on and NS/AGC off. | Useful stress/qualitative replay. Missing original render reference, actual phase annotations and controlled geometry/volume; cannot recover a calibrated Pi AEC or acoustic latency benchmark from them. |
| Generated controls and derivatives | Seven top-level WAVs; `readiness-processed.wav` duplicates synthetic speech content. Keyword variants, perturbations, provider playback and converted audio also exist under experiments/results. | Engineering fixtures and historical output, not additional independent human takes. Do not count artificial browser-test captures as live speech. |

Top-level recordings:64 audio files (57 WebM,7 WAV),63 byte-distinct hashes. Across the three inventoried roots:188 audio files,164 hashes. Those totals include converted, generated, playback and fake-test artifacts; **164 does not mean164 human recordings**. This inventory does not establish audible content, absence of clipping or Pi provenance. No paired raw/processed/reference Pi session was identified in these roots; recordings still located only on Pi must be inventoried there before asking for new captures.

A ready playback reference was subsequently prepared offline from the existing18s reading: `var/experiments/022-native-voice-duplex/reference-reading-16k.wav` (mono PCM16/16k,18.0s), with source/output hashes in `reference-preparation.json`. This is a new derivative after the inventory snapshot, not an extra human take. Transfer it privately to Pi if needed; it is deliberately absent from GitHub. No inference or new voice recording was needed. It supplies a known reference for new trials; it cannot reconstruct historical AEC reference audio.

## Reuse authorization

Original six guided manifests historically record project/university replay consent and `other_providers:false`. Preserve those manifests unchanged. **The user explicitly authorized reuse of existing project recordings with new services needed for later testing on 2026-11-09.** That later instruction authorizes relevant service-evaluation uploads despite the older restriction. Do not infer paid-call approval or permission to publish recordings on GitHub. No service was contacted in this inventory. Record the later authorization in each evaluation manifest; separate future phase/provider/cost budgets still apply.

## What is still needed from the next Pi visit

Do not repeat all54 guided prompts or request a new general laptop recording set. Follow the supplemental acquisition commands in PHASE2_REPORT.md; they retain the same physical setup and actual raw/processed streams.

| Priority | Capture once | Why existing files do not cover it |
|---|---|---|
| Required first | Inventory audio already on Pi; exact devices/routing/plugin/dependency versions, geometry, distance, volume, room/fan conditions and actual reference WAV/hash. | Local browser recordings cannot describe Pi hardware or render-reference wiring. |
| Required | 2–3min quiet Pi speech: representative existing phrases, brief/long pauses, numbers/names, stop-speaking versus cancel-task; a few novel sentences. Add near/far and normal/soft labels. | Isolates the actual mic/room path. No need to reread all suites. Record the words actually spoken, including hesitations, rather than assuming prompt compliance. |
| Required | Approximately2min echo-only and2min sustained human/reference overlap at normal geometry; repeat at another intended volume/distance if time permits. Save original reference, raw source and processed source. | The old laptop pair lacks a reference and is a different acoustic system. Continuous-overlap corpus mode disables automatic interruption deliberately; it is not proof of barge-in. |
| Required | Repeated actual interruption trials:50 for the existing p95 gate, using varied short/long/soft onsets; annotate false stops and lost initial words. A smaller batch is useful but cannot close the existing gate. | Isolated stop-command utterances do not contain a simultaneous speaker, actual onset or audible-stop evidence. |
| Required for wake qualification | >=100 naturally spoken wake examples across intended conditions plus confusable phrases/noise, and >=2h negative-room exposure for the existing rate gate. | A few wake clips/generated phrases cannot estimate reliability. Recording alone provides replay data; actual live wake integration must still be validated. |
| Required for duplex/soak gate | >=30min uninterrupted capture/render with resource logging; ensure storage headroom and preserve failures. | Short clips cannot prove device stability or concurrent Pi resource bounds. |
| Optional useful | A second consenting speaker, additional accents/rooms, independent observer recording hearing both voice and speaker. | Improves generalization and acoustic annotation; do not block primary capture on this. |

Transcripts and precise acoustic annotations can be completed offline after capture; immediately retain intended phrase IDs, condition notes and the recordings. Do not spend scarce attended time repeating intact clips just to finish labels.

Suggested order when human time is limited: basic smoke/routing → quiet speech → sustained echo/overlap → interruption batch → wake examples. Have the Pi agent complete permitted longer logging while the setup remains available. Do not change devices, volume, services or geometry without labeling a new condition. Allow roughly30–45min attended time if collecting50 fixed24s interruption trials; the30min continuity and2h negative gates are additional unless valid simultaneous protocols are explicitly documented. A shorter visit is still useful; leave unfinished gates pending.

## Offline work enabled by returning the evidence

- Confirm literal transcripts and onset/end annotations; freeze development/held-out split by original take and condition. Keep recodes/augmentations of a take in the same split.
- Replay real Pi and existing browser audio through ASR/provider adapters in separately authorized phases; compare WER, entity preservation, silence hallucination, partial/final ordering and turn segmentation.
- Test VAD/wake thresholds, endpoint policy, resampling and backlog behavior on labeled raw/processed recordings; report replay timing separately from capture/network timing.
- Exercise steering, context, approvals, retries, interruption and recovery with transcript/events replay and deterministic failure injection. These remain software tests, not proof of live model task success.
- Use known reference/raw/processed triplets to investigate residual echo and intelligibility. Replaying an old processed stream cannot predict how a different live AEC algorithm would behave; alternate AEC evaluation needs its complete reference/timing contract.

Acoustic routing, newly changed AEC/device behavior, actual audible cancellation and final end-to-end qualification still need physical confirmation. Collected evidence minimizes repeat visits; it cannot guarantee that no later hardware trial will be needed. This preparation does not authorize Phase3 or later implementation.
