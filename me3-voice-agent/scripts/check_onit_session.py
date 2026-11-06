"""One bounded Swift Session → Onit draft task trial with configured free-only model.

Run only when the project reasoning key and explicit request limit are configured.
ASR, TTS, and playback are local mocks; no microphone or device is opened.
"""

import asyncio
import argparse
import json
from pathlib import Path

from me3_voice.artifacts import output_path
from me3_voice.configuration import load_backends
from me3_voice.credentials import load_secrets
from me3_voice.onit_tasks import OnitTaskExecutor
from me3_voice.providers import MockLLM
from me3_voice.session import Session
from me3_voice.telemetry import summarize, write_trace
from me3_voice.transcript import Transcript

PYTHON = Path("var/cache/onit-venv/bin/python").resolve()
BACKENDS = Path("var/cache/voice-backends.json")
SECRETS = Path("var/cache/voice-backends.env")
TASK = (
    "agent task: create a self-contained index.html page with title Swift Mini Project, "
    "heading Raspberry Pi 5, and two paragraphs containing exactly: "
    "Raspberry Pi 5 is a single-board computer. It can run Linux. "
    "Use no scripts, stylesheets, network resources or deployment. Save only the draft."
)


class FixedSpeech:
    async def transcribe(self, key, pcm):
        return TASK

    async def synthesize(self, key, text):
        yield b"\x00\x00"  # Local protocol marker, never audible speech.


async def sink(pcm):
    return None


async def main(output, trace, backend_config=BACKENDS, secrets_file=SECRETS, llm_secrets_file=None):
    if output == trace or output.exists() or trace.exists():
        raise FileExistsError("Preserve existing Onit trial receipt/trace before another run")
    backends = load_backends(backend_config)
    selected = backends["llm"]
    if selected.request_limit is None or selected.request_limit < 2:
        raise ValueError("This one task needs an explicitly authorized two-request limit")
    credentials = load_secrets(secrets_file, {selected.auth_env})
    if llm_secrets_file:
        credentials.update(load_secrets(llm_secrets_file, {selected.auth_env}))
    tasks = OnitTaskExecutor(PYTHON, selected, credentials, max_requests=2)
    display = Transcript(
        mode="HPC-live", backend_modes={"asr": "mock", "llm": "external", "tts": "mock"}
    )
    async with Session(
        FixedSpeech(),
        MockLLM(),
        sink,
        tasks=tasks,
        transcript=display,
        turn_timeout=90,
        execution_mode="HPC-live",
    ) as session:
        # Exclude worker cold import from the task turn; measure it separately.
        tasks.prewarm()
        await tasks.startup
        session.activate("wake_injected_file")
        await (await session.submit(b"fixed-offline-input"))
        snapshot = display.snapshot()
        tool = next((m for m in session.history if m["role"] == "tool"), None)
        result = json.loads(tool["content"]) if tool else {}
        output_checks = {"artifact_present": False}
        if result.get("task_id") and "index.html" in result.get("artifacts", []):
            import hashlib
            from html.parser import HTMLParser

            body = await tasks.read_artifact(result["task_id"], "index.html")

            class InspectHTML(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.tags = []
                    self.text = []
                    self.external = False
                    self.active_attributes = False

                def handle_data(self, data):
                    self.text.append(data)

                def handle_starttag(self, tag, attrs):
                    self.tags.append(tag)
                    self.active_attributes |= any(
                        name.lower().startswith("on") for name, _ in attrs
                    ) or tag in {"iframe", "object", "embed"}
                    self.external |= any(
                        name in {"src", "href"} and value and not value.startswith("#")
                        for name, value in attrs
                    )

            inspection = InspectHTML()
            inspection.feed(body.decode("utf-8"))
            output_checks = {
                "artifact_present": True,
                "sha256": hashlib.sha256(body).hexdigest(),
                "bytes": len(body),
                "has_title": "title" in inspection.tags
                and "Swift Mini Project" in " ".join(inspection.text),
                "two_paragraphs": inspection.tags.count("p") >= 2,
                "required_content": all(
                    text in " ".join(inspection.text)
                    for text in ["Raspberry Pi 5 is a single-board computer.", "It can run Linux."]
                ),
                "has_heading": "h1" in inspection.tags,
                "no_script_or_external_dependency": "script" not in inspection.tags
                and not inspection.external
                and not inspection.active_attributes,
            }
        write_trace(trace, session.events)
        summary = summarize(session.events)
    attempt_audit = []
    worker_diagnostics = []
    audit_available = False
    for operation in tasks.operations:
        import hashlib

        audit_path = (
            tasks.root
            / hashlib.sha256(operation.encode()).hexdigest()[:24]
            / "request-attempts.jsonl"
        )
        stage_path = audit_path.parent / "worker-stages.json"
        if stage_path.is_file():
            worker_diagnostics.append(json.loads(stage_path.read_text()))
        if audit_path.is_file():
            audit_available = True
            attempt_audit.extend(json.loads(line) for line in audit_path.read_text().splitlines())
    report = {
        "kind": "onit-swift-session-trial",
        "mode": "HPC-live-task-with-mock-speech",
        "model": selected.model,
        "onit_worker_startup_ms": tasks.startup_ms,
        "onit_model_requests": result.get("requests"),
        "onit_request_attempts": sum(d["attempts"] for d in worker_diagnostics)
        if worker_diagnostics
        else len(attempt_audit)
        if audit_available
        else None,
        "worker_diagnostics": worker_diagnostics,
        "request_upper_bound": 2,
        "request_attempt_audit": attempt_audit,
        "transcript": snapshot,
        "onit_metrics": result.get("onit_metrics"),
        "task_receipts": snapshot["tasks"],
        "artifact_names": result.get("artifacts", []),
        "output_checks": output_checks,
        "turns": [
            {
                "turn_id": row["turn_id"],
                "outcome": row["outcome"],
                "durations_ms": row["durations_ms"],
            }
            for row in summary["turns"]
            if row["turn_id"] > 0
        ],
        "limitations": "Fixed mock ASR text, mock reasoning router/TTS/inaudible sink; only Onit model/tool stage is live. No browser, Pi, acoustic latency, external action or deployment.",
    }
    with output.open("x") as target:
        json.dump(report, target, indent=2)
    print(
        f"Bounded Onit task trial saved: {output}; outcome={report['turns'][0]['outcome'] if report['turns'] else 'none'}; requests={report['onit_model_requests']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="var/results/onit-session-live-trial.json")
    parser.add_argument("--trace", default="var/results/onit-session-live-trial.jsonl")
    parser.add_argument("--backend-config", type=Path, default=BACKENDS)
    parser.add_argument("--secrets-file", type=Path, default=SECRETS)
    parser.add_argument("--llm-secrets-file", type=Path)
    args = parser.parse_args()
    asyncio.run(
        main(
            output_path(args.output),
            output_path(args.trace),
            args.backend_config,
            args.secrets_file,
            args.llm_secrets_file,
        )
    )
