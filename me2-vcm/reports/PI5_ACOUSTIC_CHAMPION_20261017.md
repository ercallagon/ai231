# Pi 5 Acoustic Benchmark: Champion (2026-10-17)

## Scope

Official laptop-speaker-to-Pi-5-microphone benchmark, Champion only. Baseline was compared with Champion on frozen injected test data, not acoustically. The benchmark used 202 wake-present clips (186 supported commands, 16 OOS), 16 no-wake controls, seed 42, two pre-screened wake recordings, a two-second wake-to-command gap and an 800 ms endpoint. Echo Spot playback into the USB microphone measured 29.3 dB SNR. Screened wake recordings bias wake recall; this is not unbiased wake evaluation or unseen-speaker confirmation. Test/holdout are previously exposed.

Source files remain private in the research workspace. Their SHA-256 values are:

| File | SHA-256 |
| --- | --- |
| report.md | `a242145b96c0d653e4744c3c69bc468d94b9a4edb0ca83229607404d93c20333` |
| metrics.json | `4b7329f2da671486761c0d7de1ce8c401a64e4e257c152b654ff3718b64eb575` |
| trials.csv | `e0257615c57ae444646637b8cd917a82d0022e1ac4fbe9186ca2b62ad9e7372b` |
| pi_metrics.csv | `2d16aa110e49b487adfbcd75750417b07fb6d2f9cc250b469031adc2bd671d29` |

The export identifies the Champion profile and selected model path but records neither the executing Git revision nor an execution-time model/config hash. The profile matches the selected model, but exact loaded-byte identity is not cryptographically proven by this run. Checked-in hashes remain in `ARTIFACTS.json` and `data/model_reference.json`.

## Recognition and rejection

| Result | Count | Rate | Scope |
| --- | ---: | ---: | --- |
| Intent accuracy, wake-present | 184/202 | 91.1% | Supported and OOS trials |
| Accepted-correct supported intent + slot | 170/186 | 91.4% | Rejections/misses/wrong predictions count as failures |
| Strict supported success, real | 74/86 | 86.0% | Frozen source metadata |
| Strict supported success, synthetic | 96/100 | 96.0% | Frozen source metadata |
| OOS false accepts | 2/16 | 12.5% | 0/10 real; 2/6 synthetic |
| Supported commands rejected/missed | 16/186 | 8.6% | 9 other/OOS rejections, 1 no-sustained-speech, 6 no response |
| Exact slot value, conditional | 99/99 | 100% | Correct intent and scored slot only; not overall slot success |
| No-wake controls with wake event | 4/16 | 25.0% | All four rejected OOS; no accepted prediction/action |
| No response | 7/202 | 3.5% | Wake-present trials |

The generated benchmark report contains a contradictory sentence claiming all 10 OOS clips were real. Trial-level rows and numeric tables show 16 OOS clips: 10 real and 6 synthetic; both false accepts were synthetic. This summary follows row-level evidence. The 16 no-wake sample is too small for a stable hourly false-wake estimate. There were four multi-decision trials and one zero-audio decision; the official scorer uses the first decision in each trial interval.

Selected weaker phrase results include LIGHT_OFF 3/6, temperature-18 0/2 and temperature-22 1/2. Sample counts are small. Utterance-level rows and transcripts are not included in this public summary.

## Pi 5 performance

Raspberry Pi 5 Model B Rev 1.0, Debian 13, Python 3.13.5, ONNX Runtime CPU provider, one intra-op/inter-op thread, USB microphone `plughw:2,0`.

| Metric | Result | Meaning |
| --- | ---: | --- |
| Feature preparation + inference p95 | 116.5 ms (n=194) | Processed decisions; zero-audio excluded |
| Actual-audio RTF p95 | 0.1475 (n=194) | Processed decisions |
| Detector-estimated speech-end -> decision p50/p95 | 920.2/926.7 ms (n=194) | Includes endpoint wait; last-voiced-frame read time, not acoustic ground truth; before services |
| Model profiler parameters / ONNX / compute | 499,784 / 2.04 MB / 363 MFLOP | Profiler count differs from 488,264 learned training parameters |
| Runtime RSS max / CPU mean | 201.7 MB / 61.3% | Run-specific |
| Pi temperature max / throttling flags | 51.2 C / none observed | Run-specific |
| Wall time | 63.9 min | Full benchmark |

Do not use the benchmark's laptop-observed response p95 of 1.776 s as a clean command-end latency: nine of 195 values are negative, consistent with early/extra decisions or timestamp association. The 116.5 ms value is not speech-end latency, and the 926.7 ms detector estimate is not annotated acoustic timing or action latency.

The tested 800 ms endpoint is now the runtime default. This is not a paired 550/800/1200 ms experiment. Pauses longer than 800 ms may split commands; `--end-silence-ms 1200` remains an explicit pause-tolerance override. Complete-word retention was not manually annotated. Results do not establish performance in noise, during music/AEC, or with independent speakers.

## Decision

The evidence supports Champion as the current Pi 5 submission candidate: measured acoustic supported-command success was 91.4%, and processed feature+inference p95 was 116.5 ms. Report rejection, small denominators and timing qualifications alongside those numbers. Do not claim acoustic Baseline superiority, wake recall, unseen-speaker generalization or a valid end-to-action p95.
