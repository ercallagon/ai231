"""Reproducible mock-only LangGraph task/checkpoint and event-loop benchmark."""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import time
import uuid

from langchain_core.messages import AIMessage

from me3_voice.configuration import load_backends
from me3_voice.task_runtime import TaskRuntime


class FixtureModel:
    def __init__(self):
        self.calls = 0

    async def __call__(self, _messages, _tools):
        self.calls += 1
        if self.calls % 3 == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_artifact",
                        "args": {"path": "result.txt", "content": "fixture result"},
                        "id": f"write-{self.calls}",
                    }
                ],
            )
        if self.calls % 3 == 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_artifact",
                        "args": {"path": "result.txt"},
                        "id": f"read-{self.calls}",
                    }
                ],
            )
        return AIMessage(content="Fixture artifact was read back successfully.")


async def wait_terminal(runtime, task_id):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        status = await runtime.status(task_id)
        if status["status"] in {"succeeded", "completed_unverified", "failed", "cancelled"}:
            return status
        await asyncio.sleep(0.002)
    raise TimeoutError("benchmark task did not reach terminal state")


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def memory_snapshot():
    result = {"rss_kib": None, "pss_kib": None, "swap_kib": None}
    try:
        for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in {"Rss", "Pss", "Swap"}:
                result[{"Rss": "rss_kib", "Pss": "pss_kib", "Swap": "swap_kib"}[key]] = int(
                    rest.strip().split()[0]
                )
    except (OSError, ValueError, IndexError):
        pass
    return result


def source_provenance():
    paths = [
        Path("src/me3_voice/task_runtime.py"),
        Path("src/me3_voice/native_app.py"),
        Path("src/me3_voice/native_bridge.py"),
        Path("scripts/benchmark_phase5_runtime.py"),
    ]
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.is_file()
    }
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        project_is_repository = Path(root).resolve() == Path.cwd().resolve()
    except (OSError, subprocess.CalledProcessError):
        root, commit, project_is_repository = None, None, False
    return {
        "parent_git_root": root,
        "parent_git_head": commit,
        "me3_project_has_own_git_root": project_is_repository,
        "me3_workspace_state": "untracked_in_parent_repository",
        "source_sha256": hashes,
    }


async def run(samples, database, artifacts):
    backend = load_backends()["llm"]
    model = FixtureModel()
    runtime = TaskRuntime(
        backend,
        database=database,
        artifact_root=artifacts,
        model=model,
    )
    loop_lags = []
    heartbeat_stop = asyncio.Event()

    async def heartbeat():
        interval = 0.005
        expected = time.perf_counter() + interval
        while not heartbeat_stop.is_set():
            await asyncio.sleep(interval)
            now = time.perf_counter()
            loop_lags.append(max(0.0, now - expected))
            expected = now + interval

    started = time.perf_counter()
    await runtime.start_runtime()
    startup_seconds = time.perf_counter() - started
    beat = asyncio.create_task(heartbeat())
    try:
        # One warm-up is kept separate from the steady-state distribution.
        first = await runtime.start("Fixture: write, read back, report", request_id="phase5-warmup")
        warmup_result = await wait_terminal(runtime, first["task_id"])
        assert warmup_result["status"] == "succeeded", warmup_result
        runtime_times = []
        per_task_checkpoint = []
        cpu_before = resource.getrusage(resource.RUSAGE_SELF)
        for index in range(samples):
            checkpoint_start = len(runtime.metrics["checkpoint_seconds"])
            task_started = time.perf_counter()
            task = await runtime.start(
                "Fixture: write, read back, report", request_id=f"phase5-sample-{index}"
            )
            final = await wait_terminal(runtime, task["task_id"])
            runtime_times.append(time.perf_counter() - task_started)
            assert final["status"] == "succeeded", final
            per_task_checkpoint.append(
                runtime.metrics["checkpoint_seconds"][checkpoint_start:]
            )
        cpu_after = resource.getrusage(resource.RUSAGE_SELF)
        await asyncio.sleep(0.02)
        checkpoint_values = [value for batch in per_task_checkpoint for value in batch]
        return {
            "evidence": "repository_local_mock_offline",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "python": os.sys.version.split()[0],
                "platform": os.uname().machine,
                "pid": os.getpid(),
                "process_count_in_benchmark": 1,
                "framework": "LangGraph StateGraph + ToolNode + AsyncSqliteSaver",
            },
            "source": source_provenance(),
            "provider": {"name": "fixture", "model": "ScriptedModel", "requests": 0},
            "workload": {
                "samples": samples,
                "warmup_samples": 1,
                "calls_per_sample": 3,
                "tools_per_sample": 2,
                "result": "isolated scratch write and readback",
            },
            "startup_seconds": startup_seconds,
            "task_seconds": {
                "p50": statistics.median(runtime_times),
                "p95": percentile(runtime_times, 0.95),
                "samples": len(runtime_times),
                "values": runtime_times,
                "checkpoint_write_seconds_by_task": per_task_checkpoint,
            },
            "checkpoint_io_seconds": {
                "p50": statistics.median(checkpoint_values),
                "p95": percentile(checkpoint_values, 0.95),
                "samples": len(checkpoint_values),
                "values": checkpoint_values,
                "task_checkpoint_p50_samples": statistics.median(
                    [len(values) for values in per_task_checkpoint]
                ),
            },
            "event_loop_lag_seconds": {
                "p50": statistics.median(loop_lags),
                "p95": percentile(loop_lags, 0.95),
                "max": max(loop_lags),
                "samples": len(loop_lags),
            },
            "cpu_seconds": {
                "user": cpu_after.ru_utime - cpu_before.ru_utime,
                "system": cpu_after.ru_stime - cpu_before.ru_stime,
            },
            "memory": memory_snapshot(),
            "runtime_counts": {
                "model_calls": runtime.metrics["model_calls"],
                "tool_calls": runtime.metrics["tool_calls"],
                "graph_invocations": runtime.metrics["graph_invocations"],
            },
            "limitations": [
                "Mock model and local scratch tools only; no provider, voice, MCP subprocess, Pi, or acoustic measurement.",
                "Event-loop lag is from this x86_64/HPC environment and is not a Pi estimate.",
            ],
        }
    finally:
        heartbeat_stop.set()
        await beat
        await runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("var/results/architecture_modernization/phase5-runtime.json"),
    )
    args = parser.parse_args()
    if not 3 <= args.samples <= 100:
        parser.error("--samples must be from 3 through 100")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    database = Path("var/cache") / f"phase5-benchmark-{token}" / "tasks.sqlite"
    artifacts = Path("var/scratch") / f"phase5-benchmark-{token}"
    result = asyncio.run(run(args.samples, database, artifacts))
    with args.output.open("w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("Wrote mock-only runtime benchmark:", args.output)


if __name__ == "__main__":
    main()
