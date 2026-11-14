"""Offline subprocess fixture: parent SIGKILLs at a durable model/effect boundary."""

import asyncio
from pathlib import Path
import sys
import time

from langchain_core.messages import AIMessage
from me3_voice.configuration import load_backends
from me3_voice.task_runtime import TaskRuntime


async def main(root, mode):
    root = Path(root)

    async def model(messages, tools):
        if mode == "workspace":
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_artifact",
                        "args": {
                            "path": "site/index.html",
                            "content": "<!doctype html><h1>Crash safe</h1>",
                        },
                        "id": "site-write-1",
                    }
                ],
            )
        if mode == "model":
            (root / "ready").write_text("model")
            await asyncio.Event().wait()
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "request_external_action",
                    "args": {"action": "publish", "details": "mock only"},
                    "id": "effect-1",
                }
            ],
        )

    async def effect(*args):
        (root / "ready").write_text("effect")
        await asyncio.Event().wait()

    runtime = TaskRuntime(
        load_backends()["llm"],
        database=root / "tasks.sqlite",
        artifact_root=root / "artifacts",
        model=model,
        effect_dispatcher=effect,
    )
    if mode == "workspace":
        # The runtime store is initialized by start_runtime, so install this
        # deterministic crash window immediately after that initialization.
        await runtime.start_runtime()
        original_read = runtime.store._read_private_file

        def crash_after_atomic_replace(path, *, dir_fd=None):
            if Path(path).name == "index.html":
                (root / "ready").write_text("workspace")
                time.sleep(60)
            return original_read(path, dir_fd=dir_fd)

        runtime.store._read_private_file = crash_after_atomic_replace
    task = await runtime.start("offline crash recovery", request_id="crash")
    (root / "task_id").write_text(task["task_id"])
    if mode == "effect":
        while True:
            status = await runtime.status(task["task_id"])
            if status["status"] == "waiting_approval":
                approval = status["approvals"][0]
                await runtime.approve(task["task_id"], approval["approval_id"], 1, "approve")
                break
            await asyncio.sleep(0.01)
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
