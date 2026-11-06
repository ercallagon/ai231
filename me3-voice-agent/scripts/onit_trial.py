"""Isolated installed-Onit reasoning smoke trial; no default CLI/tools/config loading."""

import argparse
import asyncio
import getpass
import importlib
import json
import logging
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from me3_voice.credentials import load_secrets, write_secrets  # noqa: E402
from me3_voice.configuration import ConfigurationError, load_backends  # noqa: E402

MODEL = "nvidia/nemotron-3.5-lightning:free"
SECRET = ROOT / "var/cache/openrouter.env"


def configure():
    if not sys.stdin.isatty():
        raise ConfigurationError(
            "Run configure in an interactive HPC terminal for masked key entry"
        )
    if SECRET.exists():
        raise ConfigurationError("Project OpenRouter key file already exists; refusing overwrite")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            key = getpass.getpass("Project OpenRouter API key (hidden; free-only trial): ")
    except getpass.GetPassWarning:
        raise ConfigurationError("Terminal cannot hide key entry; nothing saved") from None
    write_secrets(SECRET, {"ME3_LLM_API_KEY": key})
    print("Saved owner-only project key. No inference submitted.")


async def run(output, *, evidence_mode="live"):
    import httpx
    from openai import AsyncOpenAI

    # Import the existing installed Onit reasoning loop, not its CLI, which merges
    # global settings and starts default tools. No shared installation is edited.
    module = importlib.import_module("src.model.serving.chat")
    key = load_secrets(SECRET, {"ME3_LLM_API_KEY"}).get("ME3_LLM_API_KEY")
    if not key:
        raise ConfigurationError("Project OpenRouter key missing; run configure first")
    clients, requests, statuses = [], [], []

    class Guard(httpx.AsyncBaseTransport):
        def __init__(self):
            self.inner = httpx.AsyncHTTPTransport(retries=0)

        async def handle_async_request(self, request):
            if str(request.url) != "https://openrouter.ai/api/v1/chat/completions":
                raise RuntimeError("Unexpected network route blocked")
            body = json.loads(await request.aread())
            if body.get("model") != MODEL or len(requests) >= 2:
                raise RuntimeError("Trial request/model bound reached")
            # Fail closed on paid routes, independent of the account balance.
            body["provider"] = {
                "max_price": {"prompt": 0, "completion": 0},
                "require_parameters": True,
                "allow_fallbacks": False,
            }
            for field in (
                "top_k",
                "min_p",
                "repetition_penalty",
                "presence_penalty",
                "chat_template_kwargs",
            ):
                body.pop(field, None)
            body["max_tokens"] = min(body.get("max_tokens", 512), 512)
            requests.append(time.monotonic())
            safe = httpx.Request(
                "POST", request.url, headers=request.headers, content=json.dumps(body).encode()
            )
            # Recompute the body length after adding routing restrictions.
            safe.headers["Content-Length"] = str(len(safe.content))
            response = await self.inner.handle_async_request(safe)
            statuses.append(response.status_code)
            return response

        async def aclose(self):
            await self.inner.aclose()

    def client(**kwargs):
        kwargs.update(
            max_retries=0,
            http_client=httpx.AsyncClient(transport=Guard(), trust_env=False, timeout=30),
        )
        instance = AsyncOpenAI(**kwargs)
        clients.append(instance)
        return instance

    original = module.AsyncOpenAI
    module.AsyncOpenAI = client
    started = time.monotonic()
    result, error = None, None
    # No credentials, user files, or prior conversations in the fixed prompt.
    try:
        async with asyncio.timeout(60):
            result = await module.chat(
                host="https://openrouter.ai/api/v1",
                host_key=key,
                model=MODEL,
                instruction="A meeting starts at 14:35 and lasts 50 minutes. What time does it end? Give the answer and one short explanation.",
                timeout=30,
                safety_queue=asyncio.Queue(),
                max_api_retries=1,
                stream=True,
                think=False,
                max_tokens=512,
                max_context_tokens=16384,
                max_chat_iterations=2,
                max_final_continuations=0,
                verify_answers=False,
                harness_tools=False,
                code_execution=False,
                result_store=False,
                data_path=str(ROOT / "var/scratch/onit-trial"),
            )
    except Exception as exc:
        error = type(exc).__name__
    finally:
        module.AsyncOpenAI = original
        for instance in clients:
            await instance.close()
    report = {
        "kind": "onit-reasoning-smoke",
        "model": MODEL,
        "mode": evidence_mode if requests else "not-executed",
        "requests": len(requests),
        "http_statuses": statuses,
        "elapsed_seconds": time.monotonic() - started,
        "answer": result[:4000] if isinstance(result, str) else None,
        "error_type": error or (None if result else "NoAnswer"),
        "limitations": "Installed Onit chat loop only; no MCP task tools, speech connection, physical effects or Pi verification.",
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(f"Trial receipt: {output}; requests={len(requests)}; answer_present={bool(result)}")


async def run_dgx(output, backend_config, secrets_file, *, tool_smoke=False):
    """Call the installed Onit chat loop against ME3's configured DGX Qwen."""
    import httpx
    from openai import AsyncOpenAI

    backends = load_backends(backend_config)
    spec = backends["llm"]
    expected_endpoint = "http://202.92.159.240:8001/v1/chat/completions"
    expected_model = "Qwen/Qwen3.8-27B"
    if (
        spec.backend != "external"
        or spec.protocol != "chat-completions-v1"
        or spec.endpoint != expected_endpoint
        or spec.model != expected_model
        or not spec.allow_private_http
        or not spec.allow_live
    ):
        raise ConfigurationError(
            "DGX smoke requires the documented active DGX3 Qwen profile; no request was sent"
        )
    key = load_secrets(secrets_file, {spec.auth_env}).get(spec.auth_env)
    if not key:
        raise ConfigurationError("Configured DGX reasoning credential is missing")

    # This is a one-request connection/response smoke, not a task-capability test.
    module = importlib.import_module("src.model.serving.chat")
    clients, requests, statuses = [], [], []
    probe_tool_calls = []

    class Guard(httpx.AsyncBaseTransport):
        def __init__(self):
            self.inner = httpx.AsyncHTTPTransport(retries=0)

        async def handle_async_request(self, request):
            if str(request.url) != expected_endpoint or request.method != "POST":
                raise RuntimeError("Unexpected Onit route blocked")
            request_ceiling = 3 if tool_smoke else 1
            if len(requests) >= request_ceiling:
                raise RuntimeError("DGX smoke request ceiling reached")
            body = json.loads(await request.aread())
            if body.get("model") != expected_model:
                raise RuntimeError("Unexpected Onit model blocked")
            # The class-hosted vLLM contract is documented chat-completions;
            # omit Onit's extra sampling fields and never add OpenRouter routing.
            for field in (
                "top_k",
                "min_p",
                "repetition_penalty",
                "presence_penalty",
                "chat_template_kwargs",
            ):
                body.pop(field, None)
            if tool_smoke and not requests:
                # This smoke verifies Onit's dispatch/return path, not whether
                # the model autonomously selects tools from natural language.
                body["tool_choice"] = "required"
            body["max_tokens"] = min(body.get("max_tokens", 256), 256)
            safe = httpx.Request(
                "POST", request.url, headers=request.headers, content=json.dumps(body).encode()
            )
            safe.headers["Content-Length"] = str(len(safe.content))
            requests.append(time.monotonic())
            response = await self.inner.handle_async_request(safe)
            statuses.append(response.status_code)
            return response

        async def aclose(self):
            await self.inner.aclose()

    def client(**kwargs):
        kwargs.update(
            max_retries=0,
            http_client=httpx.AsyncClient(
                transport=Guard(), trust_env=False, timeout=spec.timeout_seconds
            ),
        )
        instance = AsyncOpenAI(**kwargs)
        clients.append(instance)
        return instance

    original = module.AsyncOpenAI
    module.AsyncOpenAI = client
    started = time.monotonic()
    result, error = None, None
    registry = None
    if tool_smoke:
        from src.type.tools import ToolRegistry

        class ProbeTool:
            url = "swift-local-inert-probe"
            tool_item = {
                "type": "function",
                "function": {
                    "name": "confirm_probe",
                    "description": "Return a fixed confirmation for this offline-safe integration check.",
                    "parameters": {
                        "type": "object",
                        "properties": {"token": {"type": "string", "enum": ["swift-check"]}},
                        "required": ["token"],
                        "additionalProperties": False,
                    },
                },
            }

            def get_tool(self):
                return self.tool_item

            async def __call__(self, **kwargs):
                if kwargs != {"token": "swift-check"}:
                    return "Rejected: invalid probe arguments."
                probe_tool_calls.append("confirm_probe")
                return "Inert probe confirmed; no project or external state changed."

        registry = ToolRegistry()
        registry.register(ProbeTool())
    try:
        async with asyncio.timeout(spec.timeout_seconds + 10):
            result = await module.chat(
                host=expected_endpoint.removesuffix("/chat/completions"),
                host_key=key,
                model=expected_model,
                instruction=(
                    "Call confirm_probe exactly once with token swift-check. Then give one concise "
                    "sentence stating whether it succeeded. Do not claim any other action."
                    if tool_smoke
                    else "In one concise sentence, explain what a Raspberry Pi 5 is. Do not use tools."
                ),
                timeout=spec.timeout_seconds,
                safety_queue=asyncio.Queue(),
                # Onit's loop requires one retry slot for its initial attempt;
                # the transport guard independently bounds this smoke.
                max_api_retries=1,
                stream=False,
                think=False,
                max_tokens=256,
                max_context_tokens=16384,
                max_chat_iterations=3 if tool_smoke else 1,
                max_final_continuations=0,
                verify_answers=False,
                harness_tools=False,
                code_execution=False,
                result_store=False,
                data_path=str(ROOT / "var/scratch/onit-dgx-smoke"),
                tool_registry=registry,
            )
    except Exception as exc:
        error = type(exc).__name__
    finally:
        module.AsyncOpenAI = original
        for instance in clients:
            await instance.close()

    report = {
        "kind": "onit-dgx-qwen-smoke",
        "smoke_type": "forced-inert-tool-loop" if tool_smoke else "plain-reasoning",
        "result": "passed"
        if isinstance(result, str) and (not tool_smoke or probe_tool_calls == ["confirm_probe"])
        else "tool_not_invoked"
        if tool_smoke and isinstance(result, str)
        else "failed",
        "model": expected_model,
        "endpoint": expected_endpoint,
        "mode": "HPC-live" if requests else "not-executed",
        "requests": len(requests),
        "http_statuses": statuses,
        "probe_tool_calls": probe_tool_calls,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        "answer": result[:2000] if isinstance(result, str) else None,
        "error_type": error or (None if result else "NoAnswer"),
        "limitations": (
            "Direct installed-Onit model/tool-loop smoke only; confirm_probe is an inert "
            "in-process stub, not an MCP server. No voice session, task effect, interruption, or Pi verification."
        ),
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(
        f"DGX Onit smoke receipt: {output}; requests={len(requests)}; answer_present={bool(result)}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["configure", "run", "run-dgx", "run-dgx-tools"])
    parser.add_argument("--output", default="var/results/onit-reasoning-smoke.json")
    parser.add_argument("--backend-config", default="var/cache/voice-backends.json")
    parser.add_argument("--secrets-file", default="var/cache/dgx3-llm.env")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    try:
        if args.command == "configure":
            configure()
        elif args.command in {"run-dgx", "run-dgx-tools"}:
            from me3_voice.artifacts import output_path

            output = output_path(args.output)
            if output.exists():
                raise ConfigurationError("Preserve existing receipt; choose a fresh output path")
            asyncio.run(
                run_dgx(
                    output,
                    Path(args.backend_config),
                    Path(args.secrets_file),
                    tool_smoke=args.command == "run-dgx-tools",
                )
            )
        else:
            from me3_voice.artifacts import output_path

            output = output_path(args.output)
            if output.exists():
                raise ConfigurationError(
                    "Preserve existing receipt; review it before another trial"
                )
            asyncio.run(run(output))
    except (ConfigurationError, ImportError) as exc:
        parser.error(
            str(exc)
            if isinstance(exc, ConfigurationError)
            else "Use the existing Onit Python environment for run"
        )


if __name__ == "__main__":
    main()
