"""Selected ME2 Bluetooth service logic, adapted into ME3 with no ME2 runtime imports.

PipeWire telephony and BlueZ MAP; real actions require explicit private configuration.
No bus connection or physical effect occurs on import. Original service semantics
were inspected; successful MAP transfer is not carrier delivery confirmation.
"""

import asyncio
import argparse
import getpass
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


class Communications:
    """Fixed demo recipients only; never automatically retry a submitted effect."""

    def __init__(self, *, enabled=False, phone_address="", call_number="", message_number=""):
        self.enabled = enabled
        model = Path("/proc/device-tree/model")
        self.on_pi = model.exists() and "raspberry pi" in model.read_text().casefold()
        self.call_number, self.message_number = call_number, message_number
        self.caller = self.sms = None
        if enabled:
            validate_number(call_number)
            validate_number(message_number)
            self.caller = BluetoothCallingBackend()
            self.sms = BluetoothMessagingBackend(phone_address, timeout=10, total_timeout=30)

    async def execute(self, action):
        if not self.enabled or not self.on_pi:
            return {
                "mode": "reported",
                "result_status": "unavailable",
                "summary": "Phone actions are not configured. Pair the demo phone and configure the demo recipients first.",
                "display": "Phone unavailable",
            }
        try:
            if action == "call_demo":
                await asyncio.to_thread(self.caller.dial, self.call_number)
                summary = (
                    "Call requested on the paired phone. An answered call is not yet confirmed."
                )
            elif action == "message_demo":
                await self.sms._operate(self.message_number)
                summary = "The demo SMS transfer completed on the paired phone. Recipient delivery is not confirmed."
            else:
                raise ValueError("unsupported communication action")
            return {
                "mode": "reported",
                "result_status": "reported",
                "summary": summary,
                "display": "Requested",
            }
        except (CallingUncertain, MessagingUncertain):
            return {
                "mode": "reported",
                "result_status": "unknown",
                "summary": "The phone request outcome is unknown. Check the phone before asking again; it was not retried.",
                "display": "Check phone",
            }
        except (CallingUnavailable, MessagingUnavailable, OSError):
            return {
                "mode": "reported",
                "result_status": "unavailable",
                "summary": "The paired phone service is unavailable. Check Bluetooth, phone permissions and the Pi session.",
                "display": "Phone unavailable",
            }


DEMO_MESSAGE = "Hello from Swift! This is the configured Raspberry Pi voice-assistant demo message."


class CallingUnavailable(Exception):
    pass


class CallingUncertain(CallingUnavailable):
    """Dial may have reached the phone; never retry automatically."""


class BluetoothCallingBackend:
    def __init__(self, *, runner=subprocess.run, timeout=15):
        self._runner = runner
        self.timeout = timeout

    def _call(self, path, interface, method, *args):
        try:
            result = self._runner(
                [
                    "busctl",
                    "--user",
                    "--json=short",
                    "call",
                    "org.pipewire.Telephony",
                    path,
                    interface,
                    method,
                    *args,
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout,
            )
            return json.loads(result.stdout) if result.stdout.strip() else {}
        except subprocess.TimeoutExpired as exc:
            if method == "Dial":
                error = CallingUncertain("Check your phone before calling again.")
                error.gateway = path
                raise error from exc
            raise CallingUnavailable("Bluetooth calling response timed out.") from exc
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise CallingUnavailable(
                "Bluetooth calling is unavailable. Check the tablet connection."
            ) from exc

    def dial(self, number, *, before_dial=None):
        if not re.fullmatch("\\+[1-9][0-9]{7,14}", number):
            raise ValueError("Configure the recipient in international format, e.g. +639...")
        modems = self._call("/org/pipewire/Telephony", "org.ofono.Manager", "GetModems")
        data = modems.get("data", [])
        gateways = data[0] if isinstance(data, list) and len(data) == 1 else data
        if not isinstance(gateways, dict) or len(gateways) != 1:
            raise CallingUnavailable("Connect exactly one calling tablet before dialing.")
        gateway = next(iter(gateways))
        if self.has_active_calls(gateway):
            raise CallingUnavailable("A call already exists. End it on the tablet first.")
        if before_dial:
            before_dial()
        response = self._call(gateway, "org.ofono.VoiceCallManager", "Dial", "s", number)
        returned = response.get("data", [])
        call_path = returned[0] if isinstance(returned, list) and len(returned) == 1 else returned
        if not isinstance(call_path, str) or not call_path.startswith(gateway + "/"):
            call_path = None
        return {
            "intent": "CALL",
            "status": "dialing",
            "message": "Call requested.",
            "data": {"gateway": gateway, "call_path": call_path},
        }

    def get_call_states(self, gateway):
        """Return only object identity/state; never expose recipient or caller data."""
        calls = self._call(gateway, "org.ofono.VoiceCallManager", "GetCalls")
        data = calls.get("data", [])
        active = data[0] if isinstance(data, list) and len(data) == 1 else data
        if active == []:
            return {}
        if not isinstance(active, dict):
            raise CallingUnavailable("Unexpected Bluetooth call-state response")
        states = {}
        known = {"active", "held", "dialing", "alerting", "incoming", "waiting", "disconnected"}
        for path, properties in active.items():
            if not isinstance(path, str) or not isinstance(properties, dict):
                raise CallingUnavailable("Unexpected Bluetooth call properties")
            state = properties.get("State")
            if isinstance(state, dict) and state.get("type") == "s":
                state = state.get("data")
            if isinstance(state, list) and len(state) == 1:
                state = state[0]
            states[path] = state if isinstance(state, str) and state in known else "unknown"
        return states

    def has_active_calls(self, gateway):
        return any((state != "disconnected" for state in self.get_call_states(gateway).values()))


class MessagingUnavailable(Exception):
    def __init__(self, message="Messaging unavailable.", *, diagnostic=None):
        super().__init__(message)
        self.diagnostic = diagnostic or {}


class MessagingUncertain(MessagingUnavailable):
    """A submission was attempted; check the tablet before repeating it."""


async def create_map_session(call, address, variant, *, retry_budget=None):
    """Retry only an explicit missing method, before any payload/send exists."""
    for attempt in range(5):
        try:
            return (
                await call(
                    "/org/bluez/obex",
                    "org.bluez.obex.Client1",
                    "CreateSession",
                    "sa{sv}",
                    [address, {"Target": variant("s", "map")}],
                )
            )[0]
        except MessagingUnavailable as exc:
            if (
                exc.diagnostic.get("dbus_error") != "org.freedesktop.DBus.Error.UnknownMethod"
                or exc.diagnostic.get("method") != "CreateSession"
            ):
                raise
            if attempt == 4:
                exc.diagnostic["session_attempts"] = 5
                raise
            if retry_budget is not None and retry_budget() <= 0.5:
                raise asyncio.TimeoutError("Bluetooth session response budget exceeded") from exc
            await asyncio.sleep(0.5)


def validate_number(number):
    if not re.fullmatch("\\+[1-9][0-9]{7,14}", number):
        raise ValueError("Set ME3 configured SMS recipient in international format, e.g. +639...")


def make_bmessage(number, text=DEMO_MESSAGE):
    """MAP UTF-8 GSM bMessage. LENGTH includes the complete MSG block."""
    validate_number(number)
    if not text.strip() or any((c in text for c in "\r\n")):
        raise ValueError("Message must be nonempty single-line text.")
    block = f"BEGIN:MSG\r\n{text}\r\nEND:MSG\r\n"
    header = f"BEGIN:BMSG\r\nVERSION:1.0\r\nSTATUS:UNREAD\r\nTYPE:SMS_GSM\r\nFOLDER:telecom/msg/outbox\r\nBEGIN:BENV\r\nBEGIN:VCARD\r\nVERSION:2.1\r\nN:;\r\nTEL:{number}\r\nEND:VCARD\r\nBEGIN:BBODY\r\nCHARSET:UTF-8\r\nLENGTH:{len(block.encode('utf-8'))}\r\n"
    return (header + block + "END:BBODY\r\nEND:BENV\r\nEND:BMSG\r\n").encode("utf-8")


class BluetoothMessagingBackend:
    """Keep one user D-Bus connection alive through session and transfer cleanup."""

    def __init__(self, address, *, timeout=45, total_timeout=None):
        if not re.fullmatch("(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}", address):
            raise ValueError(
                "Set ME3 configured phone Bluetooth address to the paired tablet Bluetooth address."
            )
        self.address = address
        self.timeout = timeout
        self.total_timeout = total_timeout

    async def _operate(self, number=None, text=DEMO_MESSAGE):
        try:
            from dbus_next import Message, MessageType, Variant
            from dbus_next.aio import MessageBus
        except ImportError as exc:
            raise MessagingUnavailable("Install dbus-next in the Pi Python environment.") from exc
        bus = None
        session = None
        submitted = False
        stage = "connect_bus"
        transfer_events = {}
        operation_deadline = (
            asyncio.get_running_loop().time() + self.total_timeout
            if self.total_timeout is not None
            else None
        )

        def remaining_budget():
            if operation_deadline is None:
                return self.timeout
            return max(
                0.001, min(self.timeout, operation_deadline - asyncio.get_running_loop().time())
            )

        def on_message(message):
            if (
                message.message_type == MessageType.SIGNAL
                and message.interface == "org.freedesktop.DBus.Properties"
                and (message.member == "PropertiesChanged")
                and (message.body[0] == "org.bluez.obex.Transfer1")
                and (session is not None)
                and message.path.startswith(session + "/")
            ):
                transfer_events.setdefault(message.path, {}).update(message.body[1])

        async def call(path, interface, member, signature="", body=None):
            reply = await asyncio.wait_for(
                bus.call(
                    Message(
                        destination="org.bluez.obex",
                        path=path,
                        interface=interface,
                        member=member,
                        signature=signature,
                        body=body or [],
                    )
                ),
                min(self.timeout, 2) if member == "RemoveSession" else remaining_budget(),
            )
            if reply.message_type == MessageType.ERROR:
                raise MessagingUnavailable(
                    "Bluetooth messaging request failed.",
                    diagnostic={"dbus_error": reply.error_name, "method": member},
                )
            return reply.body

        try:
            bus = await asyncio.wait_for(MessageBus().connect(), remaining_budget())
            bus.add_message_handler(on_message)
            match_reply = await asyncio.wait_for(
                bus.call(
                    Message(
                        destination="org.freedesktop.DBus",
                        path="/org/freedesktop/DBus",
                        interface="org.freedesktop.DBus",
                        member="AddMatch",
                        signature="s",
                        body=[
                            "type='signal',sender='org.bluez.obex',interface='org.freedesktop.DBus.Properties',member='PropertiesChanged'"
                        ],
                    )
                ),
                remaining_budget(),
            )
            if match_reply.message_type == MessageType.ERROR:
                raise MessagingUnavailable("Unable to monitor Bluetooth transfer status.")
            stage = "create_session"
            session = await create_map_session(
                call, self.address, Variant, retry_budget=remaining_budget
            )
            stage = "supported_types"
            types = (
                await call(
                    session,
                    "org.freedesktop.DBus.Properties",
                    "Get",
                    "ss",
                    ["org.bluez.obex.MessageAccess1", "SupportedTypes"],
                )
            )[0].value
            if "SMS_GSM" not in types:
                raise MessagingUnavailable(
                    "The tablet MAP session does not advertise GSM SMS support."
                )
            if number is None:
                return {"supported_types": types}
            scratch = Path(__file__).resolve().parents[2] / "var/scratch"
            scratch.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="me3-message-", dir=scratch) as directory:
                stage = "prepare_payload"
                source = Path(directory) / "message.bmsg"
                source.write_bytes(make_bmessage(number, text))
                submitted = True
                stage = "push_message"
                transfer, properties = await call(
                    session,
                    "org.bluez.obex.MessageAccess1",
                    "PushMessage",
                    "ssa{sv}",
                    [
                        str(source),
                        "telecom/msg/outbox",
                        {
                            "Charset": Variant("s", "utf8"),
                            "Retry": Variant("b", False),
                            "Transparent": Variant("b", False),
                        },
                    ],
                )
                deadline = asyncio.get_running_loop().time() + self.timeout
                if operation_deadline is not None:
                    deadline = min(deadline, operation_deadline)
                stage = "transfer_status"
                while True:
                    properties.update(transfer_events.get(transfer, {}))
                    status = properties.get("Status")
                    status = status.value if status is not None else "queued"
                    if status == "complete":
                        return {"transfer_status": "complete", "delivery_confirmed": False}
                    if status == "error":
                        raise MessagingUncertain(
                            "Message transfer failed. Check the tablet before trying again.",
                            diagnostic={"transfer_status": "error"},
                        )
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise MessagingUncertain(
                            "Message status is unknown. Check the tablet before trying again.",
                            diagnostic={"transfer_status": status},
                        )
                    await asyncio.sleep(min(0.25, remaining))
                    try:
                        polled = (
                            await asyncio.wait_for(
                                call(
                                    transfer,
                                    "org.freedesktop.DBus.Properties",
                                    "GetAll",
                                    "s",
                                    ["org.bluez.obex.Transfer1"],
                                ),
                                max(0.01, remaining),
                            )
                        )[0]
                        properties.update(polled)
                    except MessagingUnavailable as exc:
                        if (
                            exc.diagnostic.get("dbus_error")
                            != "org.freedesktop.DBus.Error.UnknownObject"
                        ):
                            raise
                        await asyncio.sleep(0)
                        terminal = transfer_events.get(transfer, {}).get("Status")
                        if terminal is None or terminal.value not in {"complete", "error"}:
                            raise
        except (MessagingUnavailable, MessagingUncertain, OSError, asyncio.TimeoutError) as exc:
            diagnostic = {
                "stage": stage,
                "error_type": type(exc).__name__,
                **getattr(exc, "diagnostic", {}),
            }
            if submitted:
                raise MessagingUncertain(
                    "Message status is unknown. Check the tablet before trying again.",
                    diagnostic=diagnostic,
                ) from exc
            raise MessagingUnavailable(
                "Messaging is unavailable. Check Bluetooth, tablet message access, and the OBEX service.",
                diagnostic=diagnostic,
            ) from exc
        finally:
            if bus is not None:
                if session is not None:
                    try:
                        await asyncio.wait_for(
                            call(
                                "/org/bluez/obex",
                                "org.bluez.obex.Client1",
                                "RemoveSession",
                                "o",
                                [session],
                            ),
                            5,
                        )
                    except (MessagingUnavailable, OSError, asyncio.TimeoutError):
                        pass
                bus.remove_message_handler(on_message)
                bus.disconnect()

    def check(self):
        return asyncio.run(self._operate())

    def send(self, number, text):
        validate_number(number)
        return asyncio.run(self._operate(number, text))


def main():
    """Secure configuration only: no --dial or --send path in this setup command."""
    from .credentials import private_path

    parser = argparse.ArgumentParser(
        description="Configure fixed demo phone recipients without placing a call or sending SMS"
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    path = private_path(args.config)
    if not path.is_file() or path.stat().st_mode & 0o077:
        parser.error("use an existing owner-only device configuration under var/cache")
    data = json.loads(path.read_text())
    if data.get("backend") != "pi":
        parser.error("communications require a separately selected Pi device configuration")
    phone = getpass.getpass("Paired phone Bluetooth MAC (hidden): ").strip()
    call = getpass.getpass("Authorized demo call recipient, +country format (hidden): ").strip()
    sms = getpass.getpass("Authorized demo SMS recipient, +country format (hidden): ").strip()
    # Validate everything before saving; constructors do not connect to D-Bus.
    Communications(enabled=True, phone_address=phone, call_number=call, message_number=sms)
    data["communications"] = {
        "enabled": True,
        "phone_address": phone,
        "call_number": call,
        "message_number": sms,
    }
    fd, temporary = tempfile.mkstemp(prefix="phone-config-", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(
        "Pi phone configuration saved privately. No call, message, pairing or service change was performed."
    )


if __name__ == "__main__":
    main()
