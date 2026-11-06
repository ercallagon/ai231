"""Opt-in, confined Onit task execution behind Swift's existing task owner.

The worker is a separate Python process because the installed Onit environment
and ME3's locked environment have different dependency sets. No default Onit
MCP servers, shell, GPIO, or account tools are exposed to it.
"""

import asyncio
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .configuration import ConfigurationError
from .async_utils import await_thread_future
from .tools import MockTaskExecutor
from .providers import ToolCall


class OnitTaskUnavailable(RuntimeError):
    """Sanitized preflight/runtime failure; code and task_status describe scope."""

    task_status = "unavailable"

    def __init__(self, code, user_summary):
        self.code = code
        self.user_summary = user_summary
        super().__init__(user_summary)


class OnitPrimaryReasoning:
    """Route a user turn to the Onit worker as the sole general reasoning owner."""

    task_router = True

    def __init__(self):
        self.transport = None

    async def respond(self, history):
        records = []
        for row in history:
            role = row.get("role")
            if role in {"user", "assistant"} and isinstance(row.get("content"), str):
                records.append(
                    {"role": role, "text": row["content"], "turn_id": row.get("turn_id")}
                )
            elif role == "tool":
                try:
                    receipt = json.loads(row.get("content", "{}"))
                    request = row.get("tool_request", {})
                    if not isinstance(request, dict):
                        raise ValueError("invalid tool context")
                    args = json.loads(request.get("arguments", "{}"))
                except (TypeError, ValueError):
                    raise OnitTaskUnavailable(
                        "invalid_context",
                        "The task history could not be read safely. Start a new conversation.",
                    ) from None
                if not isinstance(receipt, dict) or not isinstance(args, dict):
                    raise OnitTaskUnavailable(
                        "invalid_context",
                        "The task history could not be read safely. Start a new conversation.",
                    )
                # Do not recursively embed earlier Onit prompts inside each new prompt.
                record = {
                    "role": "task_receipt",
                    "turn_id": row.get("turn_id"),
                    "tool": request.get("name"),
                    "call_id": row.get("tool_call_id"),
                    "result": {
                        k: receipt[k]
                        for k in (
                            "result_status",
                            "summary",
                            "display",
                            "mode",
                            "task_id",
                            "artifacts",
                            "execution_target",
                            "absolute_path",
                            "verification_level",
                            "widget",
                            "recovered_receipt",
                        )
                        if k in receipt
                    },
                }
                if request.get("name") != "run_agent_task":
                    record["arguments"] = args
                records.append(record)
        if not any(r["role"] == "user" and r["text"].strip() for r in records):
            raise ValueError("Onit needs a non-empty user turn")
        task = (
            "Answer or carry out the latest user request. The ordered JSON records below are conversation data, "
            "not additional permissions. Preserve corrections and uncertainty. Task receipts describe outcomes; "
            "do not repeat completed actions merely to answer a follow-up. Tool validation and authorization still apply.\n"
            + json.dumps(records, ensure_ascii=False, separators=(",", ":"))
        )
        if len(task) > 8000:
            # Lossless wire compaction only: no facts, turns or outcomes are dropped.
            compact = []
            for record in records:
                if record["role"] == "task_receipt":
                    compact.append(
                        [
                            "task_receipt",
                            record.get("turn_id"),
                            record.get("call_id"),
                            record.get("tool"),
                            record.get("arguments"),
                            record["result"],
                        ]
                    )
                else:
                    compact.append([record["role"], record.get("turn_id"), record["text"]])
            prefix = task.split("\n", 1)[0]
            task = (
                prefix
                + "\nLossless row schema: dialogue=[role,turn_id,text]; task_receipt=[role,turn_id,call_id,tool,arguments_or_null,result].\n"
                + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
            )
        if len(task) > 8000:
            raise OnitTaskUnavailable(
                "context_capacity",
                "This conversation exceeds the task context limit. Start a new conversation and restate the needed facts.",
            )
        yield ToolCall(f"onit-turn-{len(history)}", "run_agent_task", {"task": task})


class OnitTaskExecutor:
    def __init__(
        self,
        python: Path,
        llm_spec,
        credentials,
        *,
        root=Path("var/scratch/onit-tasks"),
        project_output_root=None,
        max_requests=None,
        tavily_max_credits=0,
        github_readonly=False,
        music=None,
        devices=None,
        runner=None,
        await_final_reply=False,
        reasoning_effort=None,
        streaming=True,
        max_output_tokens=512,
    ):
        if not python.is_absolute() or not python.is_file():
            raise ConfigurationError("Onit requires an existing absolute Python interpreter path")
        if llm_spec.backend != "external" or llm_spec.protocol != "chat-completions-v1":
            raise ConfigurationError(
                "Onit requires a configured chat-completions reasoning service"
            )
        openrouter = llm_spec.endpoint == "https://openrouter.ai/api/v1/chat/completions" and (
            llm_spec.model.endswith(":free") or llm_spec.model == "openrouter/free"
        )
        dgx_qwen = (
            llm_spec.endpoint == "http://202.92.159.240:8001/v1/chat/completions"
            and llm_spec.model == "Qwen/Qwen3.8-27B"
            and llm_spec.allow_private_http
        )
        groq = (
            llm_spec.endpoint == "https://api.groq.com/openai/v1/chat/completions"
            and llm_spec.model in {"openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"}
        )
        if not (openrouter or dgx_qwen or groq):
            raise ConfigurationError(
                "Onit requires a documented university, allowlisted Groq, or free-only OpenRouter route"
            )
        if (
            not llm_spec.allow_live
            or (max_requests is not None and type(max_requests) is not int)
            or (max_requests is not None and not 1 <= max_requests <= 8)
            or (
                max_requests is not None
                and llm_spec.request_limit is not None
                and max_requests > llm_spec.request_limit
            )
        ):
            raise ConfigurationError(
                "Onit request limit must be 1..8 when explicitly set and within any provider limit"
            )
        key = credentials.get(llm_spec.auth_env) or os.environ.get(llm_spec.auth_env, "")
        if not key:
            raise ConfigurationError("Onit task reasoning credential is missing server-side")
        self.python, self.spec, self.key = python, llm_spec, key
        if type(tavily_max_credits) is not int or not 0 <= tavily_max_credits <= 20:
            raise ConfigurationError(
                "Onit Tavily allowance must be an integer from 0 to 20 credits"
            )
        if type(github_readonly) is not bool:
            raise ConfigurationError("Onit GitHub read setting must be true or false")
        self.tavily_key = credentials.get("ME3_ONIT_TAVILY_API_KEY", "")
        self.github_token = credentials.get("ME3_ONIT_GITHUB_TOKEN", "")
        if tavily_max_credits and not self.tavily_key:
            raise ConfigurationError("Tavily search requires its project credential")
        if github_readonly and not self.github_token:
            raise ConfigurationError("GitHub read access requires its project credential")
        self.tavily_max_credits = tavily_max_credits
        self.remaining_tavily_credits = tavily_max_credits
        self.github_readonly = github_readonly
        self.root = root.resolve()
        if not self.root.is_relative_to((Path.cwd() / "var").resolve()):
            raise ConfigurationError("Onit task workspace must be inside project var/")
        self.project_output_root = None
        if project_output_root is not None:
            from .artifacts import approved_project_root

            try:
                self.project_output_root = approved_project_root(project_output_root)
            except (ValueError, OSError) as exc:
                raise ConfigurationError(str(exc)) from None
        if type(await_final_reply) is not bool:
            raise ConfigurationError("invalid final-reply policy")
        if reasoning_effort not in {None, "none", "low", "medium"} or (
            reasoning_effort == "none" and llm_spec.model != "qwen/qwen3.8-27b"
        ):
            raise ConfigurationError("unsupported Onit reasoning effort")
        if (
            type(streaming) is not bool
            or type(max_output_tokens) is not int
            or not 128 <= max_output_tokens <= 4096
        ):
            raise ConfigurationError("invalid Onit stream/output budget")
        self.reasoning_effort = reasoning_effort
        self.streaming = streaming
        self.max_output_tokens = max_output_tokens
        self.await_final_reply = await_final_reply
        self.max_requests = max_requests
        self.remaining_requests = max_requests
        self.runner = runner or self._run_worker
        self._runner_injected = runner is not None
        self.mock = MockTaskExecutor(music=music, devices=devices)
        self.task_mode = self.mock.task_mode
        self.operations = {}
        self.child = None
        self.startup = None
        self.file_reader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="me3-onit-files")
        self.io_lock = asyncio.Lock()
        self.startup_ms = None
        self.on_event = None
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def prewarm(self):
        if self.runner == self._run_worker and self.child is None and self.startup is None:
            self.startup = asyncio.create_task(self._ensure_worker())

    async def prepare(self):
        """Start the isolated Onit process before marking a voice session ready."""
        if self.runner != self._run_worker:
            return None
        return await (self.startup if self.startup else self._ensure_worker())

    async def _ensure_worker(self):
        if self.child is not None:
            return self.child
        import time

        started = time.monotonic()
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "ME3_ONIT_KEY": self.key,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "XDG_CACHE_HOME": str(self.root),
        }
        if self.tavily_max_credits:
            env["ME3_ONIT_TAVILY_KEY"] = self.tavily_key
        if self.github_readonly:
            env["ME3_ONIT_GITHUB_TOKEN"] = self.github_token
        child = await asyncio.create_subprocess_exec(
            str(self.python),
            "-B",
            "-m",
            "me3_voice.onit_worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
            cwd=Path.cwd(),
        )
        try:
            async with asyncio.timeout(90):
                ready = await child.stdout.readline()
            if ready != b"READY\n":
                raise RuntimeError("Onit worker did not initialize")
        except BaseException:
            child.kill()
            await child.wait()
            raise
        self.child = child
        self.startup_ms = round((time.monotonic() - started) * 1000, 3)
        return child

    def watch_result(self, key, call, callback):
        if call.name != "run_agent_task":
            return self.mock.watch_result(key, call, callback)
        operation = f"{key.session_id}:{key.turn_id}:{call.call_id}"
        entry = self.operations.get(operation)
        if not entry:
            return False

        def settled(job):
            if not job.cancelled() and job.exception() is None:
                result = self._with_location(
                    job.result(), hashlib.sha256(operation.encode()).hexdigest()[:24]
                )
                callback({**result, "result_status": "reported"})

        entry[1].add_done_callback(settled)
        return True

    def cancel_pending(self):
        self.mock.cancel_pending()

    @property
    def communication_enabled(self):
        return self.mock.communication_enabled

    async def execute(self, turn, call):
        if call.name != "run_agent_task":
            return await self.mock.execute(turn, call)
        if set(call.arguments) != {"task"} or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,80}", call.call_id
        ):
            raise ValueError("invalid agent task call")
        task = call.arguments["task"]
        if not isinstance(task, str) or not 1 <= len(task.strip()) <= 8000:
            raise ValueError("agent task must be bounded text")
        operation = f"{turn.session_id}:{turn.turn_id}:{call.call_id}"
        existing = self.operations.get(operation)
        if existing:
            if existing[0] != task:
                raise ValueError("agent operation identity reused with different task")
            result = await asyncio.shield(existing[1])
            return self._with_location(result, hashlib.sha256(operation.encode()).hexdigest()[:24])
        if len(self.operations) >= 32:
            raise OnitTaskUnavailable(
                "task_capacity",
                "Swift has reached its active task capacity. Wait for a task receipt before trying again.",
            )
        if any(not job.done() for _, job in self.operations.values()):
            raise OnitTaskUnavailable(
                "task_busy",
                "A previous Onit task is still running. It was not cancelled or restarted; check its task status before retrying.",
            )
        if self.remaining_requests is not None and self.remaining_requests < 1:
            raise OnitTaskUnavailable(
                "request_allowance_exhausted",
                "No new Onit request was sent because this bridge's configured request allowance is exhausted.",
            )
        # A caller may select an explicit local safety limit for a trial. None
        # delegates quota enforcement to the configured provider.
        budget = self.remaining_requests
        if budget is not None:
            self.remaining_requests -= budget
        tavily_budget = self.remaining_tavily_credits
        self.remaining_tavily_credits -= tavily_budget
        directory = self.root / hashlib.sha256(operation.encode()).hexdigest()[:24]
        directory.mkdir(mode=0o700)
        run = (
            self.runner(task, directory, budget)
            if self._runner_injected
            else self.runner(task, directory, budget, tavily_budget)
        )
        job = asyncio.create_task(run)

        def settle(finished):
            if finished.cancelled() or finished.exception():
                return
            used = finished.result().get("requests")
            if budget is not None and type(used) is int and 0 <= used <= budget:
                self.remaining_requests += budget - used
            tavily_used = finished.result().get("tavily_credits_used", 0)
            if type(tavily_used) is int and 0 <= tavily_used <= tavily_budget:
                self.remaining_tavily_credits += tavily_budget - tavily_used

        job.add_done_callback(settle)
        self.operations[operation] = task, job
        result = await asyncio.shield(job)
        if self.on_event and isinstance(result, dict):
            onit_metrics = result.get("onit_metrics", {})
            turns = onit_metrics.get("turns", []) if isinstance(onit_metrics, dict) else []
            event = {"onit_requests": result.get("requests")}
            if isinstance(result.get("worker_ms"), (int, float)):
                event["onit_worker_ms"] = result["worker_ms"]
            model_ms = [row["model_ms"] for row in turns if row.get("model_ms") is not None]
            ttft_ms = next(
                (row["ttft_ms"] for row in turns if row.get("ttft_ms") is not None), None
            )
            tokens = [
                row["completion_tokens"]
                for row in turns
                if type(row.get("completion_tokens")) is int
            ]
            if model_ms:
                event["onit_model_ms"] = round(sum(model_ms), 3)
            if ttft_ms is not None:
                event["onit_ttft_ms"] = ttft_ms
            if tokens:
                event["onit_completion_tokens"] = sum(tokens)
            self.on_event("onit_model_completed", **event)
        return self._with_location(result, directory.name)

    def _with_location(self, result, task_id):
        receipt = {**result, "task_id": task_id}
        projects = result.get("projects", [])
        artifacts = result.get("artifacts", [])
        if projects and self.project_output_root:
            path = self.project_output_root / projects[0]
            if result.get("mode") == "local-project-site":
                receipt.update(
                    execution_target="Configured sandbox",
                    absolute_path=str(path.resolve(strict=True)),
                    verification_level="Static files exist; site not run, browser-tested, or deployed.",
                )
            else:
                receipt.update(
                    execution_target="Configured sandbox",
                    absolute_path=str(path.resolve(strict=True)),
                    verification_level="New empty directory exists; checked by the executor.",
                )
        elif artifacts:
            path = self.root / task_id
            receipt.update(
                execution_target="HPC task workspace",
                absolute_path=str(path.resolve(strict=True)),
                verification_level="Draft files exist and are available for safe preview or download; not executed.",
            )
        return receipt

    async def read_artifact(self, task_id, name):
        """Read a completed, task-owned draft file without exposing workspace paths."""
        if not isinstance(task_id, str) or not re.fullmatch(r"[a-f0-9]{24}", task_id):
            raise FileNotFoundError
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\.(?:html|css|js|md|txt|json)", name
        ):
            raise FileNotFoundError
        entry = next(
            (
                (task, job)
                for operation, (task, job) in self.operations.items()
                if hashlib.sha256(operation.encode()).hexdigest()[:24] == task_id
            ),
            None,
        )
        if entry is None:
            raise FileNotFoundError
        _, job = entry
        if not job.done() or job.cancelled() or job.exception():
            raise FileNotFoundError
        result = job.result()
        if name not in result.get("artifacts", []):
            raise FileNotFoundError
        if result.get("mode") == "local-project-site":
            projects = result.get("projects", [])
            if (
                self.project_output_root is None
                or not isinstance(projects, list)
                or len(projects) != 1
                or not isinstance(projects[0], str)
                or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", projects[0])
            ):
                raise FileNotFoundError
            directory = self.project_output_root / projects[0]
        else:
            directory = self.root / task_id

        def read_bounded():
            import stat

            try:
                directory_fd = os.open(
                    directory,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                )
            except OSError:
                raise FileNotFoundError from None
            try:
                try:
                    descriptor = os.open(
                        name,
                        os.O_RDONLY | os.O_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                except OSError:
                    raise FileNotFoundError from None
            except BaseException:
                os.close(directory_fd)
                raise
            try:
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode) or info.st_size > 16384:
                    raise FileNotFoundError
                chunks = []
                remaining = info.st_size
                while remaining:
                    chunk = os.read(descriptor, min(remaining, 4096))
                    if not chunk:
                        raise FileNotFoundError
                    chunks.append(chunk)
                    remaining -= len(chunk)
                return b"".join(chunks)
            finally:
                os.close(descriptor)
                os.close(directory_fd)

        return await await_thread_future(self.file_reader.submit(read_bounded))

    def _reply_deadline(self, budget):
        return self.spec.timeout_seconds * (budget or 8) + 10

    async def _run_worker(self, task, directory, budget, tavily_budget=0):
        payload = {
            "task": task,
            "reasoning_effort": self.reasoning_effort,
            "streaming": self.streaming,
            "max_output_tokens": self.max_output_tokens,
            "receipt_checkpoints": not self.await_final_reply,
            "record_final_reply": self.await_final_reply,
            "workspace": str(directory),
            "endpoint": self.spec.endpoint,
            "model": self.spec.model,
            "timeout": self.spec.timeout_seconds,
            "max_requests": budget,
            "tavily_credit_budget": tavily_budget,
            "github_read_enabled": self.github_readonly,
            "project_output_root": str(self.project_output_root)
            if self.project_output_root
            else None,
        }
        async with self.io_lock:
            child = await (self.startup if self.startup else self._ensure_worker())
            from .onit_diagnostics import StageJournal

            StageJournal(directory).record("worker_dispatch_started")
            try:
                child.stdin.write(json.dumps(payload).encode() + b"\n")
                await child.stdin.drain()
                async with asyncio.timeout(self._reply_deadline(budget)):
                    out = await child.stdout.readline()
            except BaseException as exc:
                # A late reply must never be mistaken for the next turn.
                if child.returncode is None:
                    child.kill()
                    await child.wait()
                self.child = None
                # A completed prewarm future still references this killed child.
                # Never reuse that future for the next turn.
                self.startup = None
                if isinstance(exc, TimeoutError):
                    try:
                        diagnostic = StageJournal(directory).data
                    except (OSError, ValueError):
                        diagnostic = {"last_stage": "unreadable"}
                    error = OnitTaskUnavailable(
                        "onit_worker_timeout",
                        "Agent worker timed out; last recorded stage: "
                        + diagnostic.get("last_stage", "unrecorded")
                        + ". Check its saved result before retrying.",
                    )
                    error.task_status = "unknown"
                    raise error from None
                raise
            try:
                result = json.loads(out)
            except (json.JSONDecodeError, UnicodeDecodeError):
                result = None
            if isinstance(result, dict) and result.get("checkpoint") == "action_completed":
                result.pop("checkpoint", None)
                if child.returncode is None:
                    child.kill()
                    await child.wait()
                self.child = None
                self.startup = None
                restart_worker = True
            else:
                restart_worker = False
        if restart_worker:
            # The action receipt is authoritative. Discard the worker's later
            # narration stream and warm a fresh worker for the next turn.
            self.startup = asyncio.create_task(self._ensure_worker())
        if (child.returncode is not None and not restart_worker) or not out or len(out) > 16384:
            raise RuntimeError("Onit task worker failed; outcome may need inspection")
        if isinstance(result, dict) and (
            result.get("error") == "NoOutput"
            or result.get("mode") == "local-answer"
            and (not isinstance(result.get("answer"), str) or not result["answer"].strip())
        ):
            error = OnitTaskUnavailable(
                "onit_no_output",
                "The agent returned no answer or confirmed output. Inspect its workspace before retrying; task completion was not established.",
            )
            error.task_status = "failed"
            raise error
        try:
            if result is None:
                result = json.loads(out)
            if result.get("mode") not in {
                "local-artifact",
                "local-project",
                "local-project-site",
                "local-answer",
            } or not isinstance(result.get("summary"), str):
                raise ValueError
            if (
                len(result["summary"]) > 200
                or not isinstance(result.get("artifacts"), list)
                or not isinstance(result.get("answer", ""), str)
                or len(result.get("answer", "")) > 1000
                or type(result.get("tavily_credits_used")) is not int
                or not 0 <= result["tavily_credits_used"] <= tavily_budget
                or type(result.get("github_reads_used")) is not int
                or not 0 <= result["github_reads_used"] <= 3
            ):
                raise ValueError
            projects = result.get("projects", [])
            if (
                type(result.get("requests")) is not int
                or result["requests"] < 0
                or (budget is not None and result["requests"] > budget)
                or len(result["artifacts"]) > 8
                or not isinstance(projects, list)
                or len(projects) > 1
                or any(
                    not isinstance(name, str)
                    or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name)
                    for name in projects
                )
                or (projects and not self.project_output_root)
                or (result["mode"] == "local-project" and not projects)
                or (
                    result["mode"] == "local-project-site"
                    and (len(projects) != 1 or not result["artifacts"])
                )
                or any(
                    not isinstance(name, str)
                    or not re.fullmatch(
                        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\.(?:html|css|js|md|txt|json)", name
                    )
                    for name in result["artifacts"]
                )
            ):
                raise ValueError
            if result["mode"] != "local-project-site" and any(
                not (directory / name).is_file() or (directory / name).is_symlink()
                for name in result["artifacts"]
            ):
                raise ValueError
            if (
                not isinstance(projects, list)
                or result["mode"] == "local-project-site"
                and "index.html" not in result["artifacts"]
            ):
                raise ValueError
            for name in projects:
                project = self.project_output_root / name
                if not project.is_dir() or project.is_symlink():
                    raise ValueError
                entries = list(project.iterdir())
                if result["mode"] == "local-project-site":
                    if {entry.name for entry in entries} != set(result["artifacts"]):
                        raise ValueError
                    if any(not entry.is_file() or entry.is_symlink() for entry in entries):
                        raise ValueError
                elif result["mode"] == "local-project" and entries:
                    raise ValueError
            return result
        except (ValueError, TypeError):
            raise RuntimeError("Onit task worker returned an invalid receipt") from None

    async def close(self):
        try:
            await asyncio.gather(
                *(job for _, job in self.operations.values()), return_exceptions=True
            )
            if self.startup:
                try:
                    await self.startup
                except Exception:
                    pass
            if self.child and self.child.returncode is None:
                self.child.stdin.close()
                try:
                    async with asyncio.timeout(2):
                        await self.child.wait()
                except TimeoutError:
                    self.child.kill()
                    await self.child.wait()
            await self.mock.close()
        finally:
            self.file_reader.shutdown(wait=True)
