"""Explicit PipeWire device peer for the canonical loopback voice bridge.

No device opens without --capture and two explicit node names. Playback ACKs
are duration-based software estimates, never acoustic completion evidence.
"""

import argparse
import asyncio
import json
import shutil
import struct
from contextlib import suppress
from pathlib import Path

import aiohttp


async def reap(process):
    if process is not None and process.returncode is None:
        with suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), 2)
        except TimeoutError:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()


class PipeWirePlayback:
    def __init__(self, target, acknowledge, error_log):
        self.target, self.acknowledge, self.error_log = target, acknowledge, error_log
        self.queue = asyncio.Queue(maxsize=8)
        self.process = self.worker = None
        self.stopping = False

    async def write(self, sequence, pcm):
        if not pcm or len(pcm) % 2 or len(pcm) > 32000:
            raise ValueError("invalid output PCM packet")
        if self.worker is None:
            self.stopping = False
            self.worker = asyncio.create_task(self.run())
        await asyncio.wait_for(self.queue.put((sequence, pcm)), 10)

    async def run(self):
        opening = asyncio.create_task(
            asyncio.create_subprocess_exec(
                "pw-play",
                f"--target={self.target}",
                "--raw",
                "--format=s16",
                "--rate=16000",
                "--channels=1",
                "-",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=self.error_log,
            )
        )
        try:
            self.process = await asyncio.shield(opening)
        except asyncio.CancelledError:
            self.process = await opening
            await reap(self.process)
            self.process = None
            raise
        try:
            while not self.stopping:
                sequence, pcm = await self.queue.get()
                try:
                    self.process.stdin.write(pcm)
                    await asyncio.wait_for(self.process.stdin.drain(), 3)
                    if self.stopping:
                        return
                    await asyncio.sleep(len(pcm) / 32000)
                    if not self.stopping:
                        await self.acknowledge(sequence)
                finally:
                    self.queue.task_done()
        finally:
            await reap(self.process)
            self.process = None

    async def stop(self):
        self.stopping = True
        # Stop the owned physical-output process before waiting for queue cleanup.
        await reap(self.process)
        if self.worker:
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
            self.worker = None
        while not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()
        await reap(self.process)


async def run(args):
    url = f"http://127.0.0.1:{args.port}"
    Path("var/logs").mkdir(parents=True, exist_ok=True)
    capture = None
    tasks = []
    with Path("var/logs/pi-audio-device.log").open("ab") as log:
        async with aiohttp.ClientSession() as client:
            async with client.get(
                url + "/api/health", timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                health = await response.json()
                if health.get("bridge") != "ready" or health.get("voice_session") != "idle":
                    raise RuntimeError("Bridge is unavailable or a voice peer already owns it")
            async with client.ws_connect(url + "/api/voice/ws", max_msg_size=65536) as ws:
                await ws.send_json(
                    {
                        "type": "hello",
                        "rate": 48000,
                        "channels": 1,
                        "capture_source": "pipewire",
                        "aec_reported": None,
                        "conversation_mode": "followup",
                    }
                )
                ready = await asyncio.wait_for(ws.receive_json(), 60)
                if ready.get("type") != "ready":
                    raise RuntimeError("Voice runtime did not become ready; inspect bridge log")

                async def acknowledge(sequence):
                    await ws.send_json({"type": "audio_ack", "id": sequence})

                output = PipeWirePlayback(args.playback_node, acknowledge, log)
                try:
                    capture = await asyncio.create_subprocess_exec(
                        "pw-record",
                        f"--target={args.capture_node}",
                        "--raw",
                        "--format=s16",
                        "--rate=48000",
                        "--channels=1",
                        "-",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=log,
                    )
                    print(
                        "Pi audio peer connected. Say Hey Swift. Ctrl+C ends capture. AEC is not qualified.",
                        flush=True,
                    )

                    async def send_input():
                        while True:
                            pcm = await asyncio.wait_for(capture.stdout.readexactly(1920), 3)
                            await ws.send_bytes(pcm)
                            if output.worker and output.worker.done():
                                output.worker.result()

                    async def receive_output():
                        async for message in ws:
                            if message.type == aiohttp.WSMsgType.BINARY:
                                if len(message.data) < 6:
                                    raise ValueError("invalid audio envelope")
                                await output.write(
                                    struct.unpack("<I", message.data[:4])[0], message.data[4:]
                                )
                            elif message.type == aiohttp.WSMsgType.TEXT:
                                item = json.loads(message.data)
                                if item.get("type") == "audio_stop":
                                    await output.stop()
                                elif item.get("type") == "error":
                                    raise RuntimeError(
                                        "Bridge reported a runtime failure; inspect its trace"
                                    )

                    tasks = [
                        asyncio.create_task(send_input()),
                        asyncio.create_task(receive_output()),
                    ]
                    done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    await output.stop()
                    await reap(capture)
                    with suppress(ConnectionError):
                        await ws.send_json({"type": "stop"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-node", required=True)
    parser.add_argument("--playback-node", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--capture", action="store_true", help="explicitly open the selected microphone and speaker"
    )
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or any(
        not value or value.startswith("-") for value in (args.capture_node, args.playback_node)
    ):
        parser.error("explicit node names and an unprivileged loopback port are required")
    missing = [name for name in ("pw-record", "pw-play") if not shutil.which(name)]
    if missing:
        parser.error("Missing existing PipeWire tools: " + ", ".join(missing))
    if not args.capture:
        print(
            "Utilities found. No devices opened. Inspect pw-record/pw-play --help and selected node names before adding --capture."
        )
        return
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
