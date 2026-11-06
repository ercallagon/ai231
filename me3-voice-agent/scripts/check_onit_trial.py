"""Offline wire check against the existing Onit environment; no model service calls."""

import sys
import asyncio
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))
import onit_trial as t
import httpx

logging.disable(logging.CRITICAL)
t.load_secrets = lambda *a, **kw: {"ME3_LLM_API_KEY": "offline-dummy-key"}
seen = []


class Peer(httpx.AsyncBaseTransport):
    def __init__(self, **kwargs):
        pass

    async def handle_async_request(self, r):
        b = json.loads(await r.aread())
        seen.append(b)
        assert b["provider"]["max_price"] == {"prompt": 0, "completion": 0}
        assert b["model"].endswith(":free") and b["max_tokens"] <= 512
        assert not {
            "presence_penalty",
            "top_k",
            "min_p",
            "repetition_penalty",
            "chat_template_kwargs",
        }.intersection(b)
        event = {
            "id": "offline",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": t.MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "15:25, fifty minutes after 14:35."},
                    "finish_reason": None,
                }
            ],
        }
        done = {**event, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                "data: "
                + json.dumps(event)
                + "\n\ndata: "
                + json.dumps(done)
                + "\n\ndata: [DONE]\n\n"
            ).encode(),
            request=r,
        )


httpx.AsyncHTTPTransport = Peer
out = Path("var/scratch/onit-offline-current.json")
if out.exists():
    out.unlink()
asyncio.run(t.run(out, evidence_mode="offline-wire-fixture"))
r = json.loads(out.read_text())
assert seen and "15:25" in (r["answer"] or ""), r
print(
    "Installed Onit reasoning loop + guarded HTTP client passed with offline SSE peer; no live inference."
)
