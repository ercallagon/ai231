"""Task adapters for the existing Swift session and reasoning-tool contract."""

from .devices import ACTION_TITLES, ACTION_VALUES, DeviceActions, MockDeviceOutput

MOCK_ACTION_TITLES = ACTION_TITLES
MOCK_ACTION_VALUES = ACTION_VALUES


def task_title(call):
    """Only allowlisted, user-facing labels enter the transcript projection."""
    if call.name == "control_music":
        return {
            "play": "Play music",
            "pause": "Pause music",
            "resume": "Resume music",
            "stop": "Stop music",
            "next": "Next track",
            "previous": "Previous track",
            "play_playlist": "Play a playlist",
            "volume_up": "Turn music up",
            "volume_down": "Turn music down",
            "volume_set": "Set music volume",
        }.get(call.arguments.get("action"), "Control music")
    if call.name == "set_mock_light":
        return (
            "Turn on the light" if call.arguments.get("enabled") is True else "Turn off the light"
        )
    if call.name == "read_mock_light":
        return "Check the light"
    if call.name in {"run_device_action", "run_mock_action"}:
        return MOCK_ACTION_TITLES.get(call.arguments.get("action"), "Device action")
    return "Background task"


class MockActions(DeviceActions):
    """Compatibility name for existing tests/scenarios; always uses mock output."""

    def __init__(self, capacity=128):
        super().__init__(output=MockDeviceOutput(), backend="mock")
        self.capacity = capacity


class MockTaskExecutor:
    """Existing task owner with a replaceable shared device adapter."""

    @property
    def communication_enabled(self):
        return bool(getattr(getattr(self.actions, "communications", None), "enabled", False))

    def __init__(self, actions=None, music=None, devices=None):
        if actions is not None and devices is not None:
            raise ValueError("pass either actions or devices, not both")
        self.actions = devices or actions or MockActions()
        self.task_mode = self.actions.mode
        from .music import MusicController

        self._owns_music = music is None
        self._owns_actions = devices is None and actions is None
        self.music = music or MusicController.configured()

    def watch_result(self, key, call, callback):
        """Observe an already-dispatched operation; never submit or retry one."""
        operation = f"{key.session_id}:{key.turn_id}:{call.call_id}"
        durable_key = call.call_id if call.call_id.startswith("ui-") else operation
        job = getattr(self, "receipt_jobs", {}).get(durable_key)
        if job is None:
            owner = self.music if call.name == "control_music" else self.actions
            entry = owner.operations.get(operation)
            job = entry[1] if entry else None
        if job is None:
            return False

        def settled(task):
            if not task.cancelled() and task.exception() is None:
                callback(task.result())

        job.add_done_callback(settled)
        return True

    def receipt_context(self, call):
        if call.name == "control_music":
            return {
                "backend": self.music.output_backend,
                "device": self.music.device_id,
                "live": bool(self.music.token),
            }
        return {"backend": self.actions.backend, "output": type(self.actions.output).__name__}

    async def execute(self, key, call):
        store = getattr(self, "receipt_store", None)
        if store is None:
            return await self._execute(key, call)
        import asyncio
        import json

        operation = f"{key.session_id}:{key.turn_id}:{call.call_id}"
        identity = call.call_id if call.call_id.startswith("ui-") else operation
        if not hasattr(self, "receipt_jobs"):
            self.receipt_jobs, self.receipt_arguments = {}, {}
        arguments = json.dumps(
            [call.name, call.arguments, self.receipt_context(call)], sort_keys=True
        )
        if identity in self.receipt_jobs:
            if self.receipt_arguments[identity] != arguments:
                raise ValueError("action identity reused with different arguments")
            return await asyncio.shield(self.receipt_jobs[identity])
        if len(self.receipt_jobs) >= 128:
            raise ValueError("receipt operation capacity reached")

        async def dispatch():
            state, result, fingerprint = await store.claim(
                identity, call, self.receipt_context(call)
            )
            if state == "pending":
                return {
                    "mode": "reported",
                    "result_status": "unknown",
                    "summary": "A prior dispatch has no confirmed outcome. It was not repeated; inspect the target before issuing a new action.",
                    "display": "Outcome unknown",
                    "recovered_receipt": True,
                }
            if state == "complete":
                # This is a historical receipt, not current device or timer state.
                result = dict(result)
                result.pop("widget", None)
                result["result_status"] = "reported"
                result["summary"] = (
                    "Recorded "
                    + ("mock " if result.get("mode") == "mock" else "")
                    + "outcome; action was not repeated. Current state and process-local timers were not restored."
                )
                result["display"] = "Recorded outcome"
                result["recovered_receipt"] = True
                return result
            from .receipts import DispatchUncertain

            try:
                result = await self._execute(key, call)
            except (TimeoutError, ConnectionError):
                raise DispatchUncertain("dispatch outcome unconfirmed") from None
            try:
                await store.finish(identity, fingerprint, result)
            except Exception:
                # The adapter result is still authoritative for this process;
                # the pending disk record conservatively blocks restart replay.
                result = dict(result)
                result["summary"] = (
                    result.get("summary", "Adapter reported an outcome.")[:110]
                    + " Receipt persistence failed; do not retry automatically."
                )
            return result

        self.receipt_arguments[identity] = arguments
        self.receipt_jobs[identity] = asyncio.create_task(dispatch())
        return await asyncio.shield(self.receipt_jobs[identity])

    async def _execute(self, key, call):
        import re

        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", call.call_id):
            raise ValueError("invalid task call identity")
        operation = f"{key.session_id}:{key.turn_id}:{call.call_id}"
        music_keys = set(call.arguments)
        music_action = call.arguments.get("action")
        valid_music_call = (
            music_action == "play_playlist"
            and music_keys in ({"action", "query"}, {"action", "query", "market"})
        ) or (music_action != "play_playlist" and music_keys in ({"action"}, {"action", "query"}))
        if call.name == "control_music" and valid_music_call:
            return await self.music.execute(
                operation,
                call.arguments.get("action"),
                call.arguments.get("query", ""),
                call.arguments.get("market"),
            )
        if call.name == "set_mock_light" and set(call.arguments) == {"enabled"}:
            if self.task_mode != "mock":
                raise ValueError("legacy mock-light action is disabled by the Pi device backend")
            result = await self.actions.set_light(operation, call.arguments["enabled"])
            result["enabled"] = self.actions.light
            return result
        if call.name == "read_mock_light" and call.arguments == {}:
            if self.task_mode != "mock":
                raise ValueError("light sensor readback is not available")
            return {
                "mode": self.actions.mode,
                "verified_in_memory": self.actions.mode == "mock",
                "enabled": self.actions.light,
            }
        if call.name in {"run_device_action", "run_mock_action"} and set(call.arguments) in (
            {"action"},
            {"action", "value"},
        ):
            action, value = call.arguments["action"], call.arguments.get("value")
            if not isinstance(action, str) or value is not None and not isinstance(value, str):
                raise ValueError("invalid device action arguments")
            return await self.actions.run(operation, action, value)
        raise ValueError("unsupported task or arguments")

    def cancel_pending(self):
        self.music.cancel_pending()

    async def close(self):
        import asyncio

        if getattr(self, "receipt_jobs", None):
            await asyncio.gather(*self.receipt_jobs.values(), return_exceptions=True)
        if getattr(self, "receipt_store", None):
            await self.receipt_store.close()
        if self._owns_music:
            await self.music.close()
        if self._owns_actions:
            await self.actions.close()
