"""Configured SMS via BlueZ MAP; no classifier, GPIO, or cloud dependency."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import tempfile
from hey_swift.integration.config import communication_values
from hey_swift.integration.config import runtime_root

from hey_swift.actions.services.information import ActionResult

DEMO_MESSAGE = ("Hello from Swift! This message was sent by voice command from my "
                "Raspberry Pi assistant. Have a great day!")


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
            return (await call("/org/bluez/obex", "org.bluez.obex.Client1",
                               "CreateSession", "sa{sv}",
                               [address, {"Target": variant("s", "map")}]))[0]
        except MessagingUnavailable as exc:
            if (exc.diagnostic.get("dbus_error") != "org.freedesktop.DBus.Error.UnknownMethod"
                    or exc.diagnostic.get("method") != "CreateSession"):
                raise
            if attempt == 4:
                exc.diagnostic["session_attempts"] = 5
                raise
            if retry_budget is not None and retry_budget() <= .5:
                raise asyncio.TimeoutError("Bluetooth session response budget exceeded") from exc
            await asyncio.sleep(0.5)


def validate_number(number):
    if not re.fullmatch(r"\+[1-9][0-9]{7,14}", number):
        raise ValueError("Set VCM_MESSAGE_NUMBER in international format, e.g. +639...")


def make_bmessage(number, text=DEMO_MESSAGE):
    """MAP UTF-8 GSM bMessage. LENGTH includes the complete MSG block."""
    validate_number(number)
    if not text.strip() or any(c in text for c in "\r\n"):
        raise ValueError("Message must be nonempty single-line text.")
    block = f"BEGIN:MSG\r\n{text}\r\nEND:MSG\r\n"
    header = ("BEGIN:BMSG\r\nVERSION:1.0\r\nSTATUS:UNREAD\r\nTYPE:SMS_GSM\r\n"
              "FOLDER:telecom/msg/outbox\r\nBEGIN:BENV\r\nBEGIN:VCARD\r\n"
              f"VERSION:2.1\r\nN:;\r\nTEL:{number}\r\nEND:VCARD\r\n"
              "BEGIN:BBODY\r\nCHARSET:UTF-8\r\n"
              f"LENGTH:{len(block.encode('utf-8'))}\r\n")
    return (header + block + "END:BBODY\r\nEND:BENV\r\nEND:BMSG\r\n").encode("utf-8")


class BluetoothMessagingBackend:
    """Keep one user D-Bus connection alive through session and transfer cleanup."""

    def __init__(self, address, *, timeout=45, total_timeout=None):
        if not re.fullmatch(r"(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}", address):
            raise ValueError("Set VCM_PHONE_MAC to the paired tablet Bluetooth address.")
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
        operation_deadline = (asyncio.get_running_loop().time() + self.total_timeout
                              if self.total_timeout is not None else None)

        def remaining_budget():
            if operation_deadline is None:
                return self.timeout
            return max(.001, min(self.timeout, operation_deadline - asyncio.get_running_loop().time()))

        def on_message(message):
            if (message.message_type == MessageType.SIGNAL
                    and message.interface == "org.freedesktop.DBus.Properties"
                    and message.member == "PropertiesChanged"
                    and message.body[0] == "org.bluez.obex.Transfer1"
                    and session is not None
                    and message.path.startswith(session + "/")):
                transfer_events.setdefault(message.path, {}).update(message.body[1])

        async def call(path, interface, member, signature="", body=None):
            reply = await asyncio.wait_for(bus.call(Message(
                destination="org.bluez.obex", path=path, interface=interface,
                member=member, signature=signature, body=body or [])),
                min(self.timeout, 2) if member == "RemoveSession" else remaining_budget())
            if reply.message_type == MessageType.ERROR:
                # Error names are useful without exposing a recipient or payload.
                raise MessagingUnavailable("Bluetooth messaging request failed.",
                    diagnostic={"dbus_error": reply.error_name, "method": member})
            return reply.body

        try:
            bus = await asyncio.wait_for(MessageBus().connect(), remaining_budget())
            bus.add_message_handler(on_message)
            # Subscribe before creating a transfer: small SMS transfers can finish
            # and disappear before the first GetAll poll.
            match_reply = await asyncio.wait_for(bus.call(Message(
                destination="org.freedesktop.DBus", path="/org/freedesktop/DBus",
                interface="org.freedesktop.DBus", member="AddMatch", signature="s",
                body=["type='signal',sender='org.bluez.obex',interface='org.freedesktop.DBus.Properties',member='PropertiesChanged'"])), remaining_budget())
            if match_reply.message_type == MessageType.ERROR:
                raise MessagingUnavailable("Unable to monitor Bluetooth transfer status.")
            stage = "create_session"
            session = await create_map_session(call, self.address, Variant, retry_budget=remaining_budget)
            stage = "supported_types"
            types = (await call(session, "org.freedesktop.DBus.Properties", "Get",
                                "ss", ["org.bluez.obex.MessageAccess1", "SupportedTypes"]))[0].value
            if "SMS_GSM" not in types:
                raise MessagingUnavailable("The tablet MAP session does not advertise GSM SMS support.")
            if number is None:
                return {"supported_types": types}
            with tempfile.TemporaryDirectory(prefix="vcm-message-") as directory:
                stage = "prepare_payload"
                source = Path(directory) / "message.bmsg"
                source.write_bytes(make_bmessage(number, text))
                # After this point failures can leave an SMS queued on the tablet.
                submitted = True
                stage = "push_message"
                transfer, properties = await call(session, "org.bluez.obex.MessageAccess1",
                    "PushMessage", "ssa{sv}", [str(source), "telecom/msg/outbox",
                    {"Charset": Variant("s", "utf8"), "Retry": Variant("b", False),
                     "Transparent": Variant("b", False)}])
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
                        raise MessagingUncertain("Message transfer failed. Check the tablet before trying again.",
                                                 diagnostic={"transfer_status": "error"})
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise MessagingUncertain("Message status is unknown. Check the tablet before trying again.",
                                                 diagnostic={"transfer_status": status})
                    await asyncio.sleep(min(0.25, remaining))
                    try:
                        polled = (await asyncio.wait_for(call(transfer,
                            "org.freedesktop.DBus.Properties", "GetAll", "s",
                            ["org.bluez.obex.Transfer1"]), max(0.01, remaining)))[0]
                        properties.update(polled)
                    except MessagingUnavailable as exc:
                        if exc.diagnostic.get("dbus_error") != "org.freedesktop.DBus.Error.UnknownObject":
                            raise
                        # Removal alone proves nothing. Only a captured terminal
                        # status permits a completed result.
                        await asyncio.sleep(0)
                        terminal = transfer_events.get(transfer, {}).get("Status")
                        if terminal is None or terminal.value not in {"complete", "error"}:
                            raise
        except (MessagingUnavailable, MessagingUncertain, OSError, asyncio.TimeoutError) as exc:
            diagnostic = {"stage": stage, "error_type": type(exc).__name__,
                          **getattr(exc, "diagnostic", {})}
            if submitted:
                raise MessagingUncertain("Message status is unknown. Check the tablet before trying again.", diagnostic=diagnostic) from exc
            raise MessagingUnavailable("Messaging is unavailable. Check Bluetooth, tablet message access, and the OBEX service.", diagnostic=diagnostic) from exc
        finally:
            if bus is not None:
                if session is not None:
                    try:
                        await asyncio.wait_for(call("/org/bluez/obex", "org.bluez.obex.Client1",
                                                    "RemoveSession", "o", [session]), 5)
                    except (MessagingUnavailable, OSError, asyncio.TimeoutError):
                        pass
                bus.remove_message_handler(on_message)
                bus.disconnect()

    def check(self):
        return asyncio.run(self._operate())

    def send(self, number, text):
        validate_number(number)
        return asyncio.run(self._operate(number, text))


class MessagingService:
    def __init__(self, backend, number):
        self.backend = backend
        self.number = number

    def execute(self, intent="MESSAGE", slot_value=None):
        if intent != "MESSAGE":
            return ActionResult(intent, "unsupported", "Unsupported messaging intent.")
        if slot_value not in (None, ""):
            return ActionResult(intent, "invalid_slot", "This command does not accept a slot value.")
        try:
            validate_number(self.number)
            data = self.backend.send(self.number, DEMO_MESSAGE)
        except MessagingUncertain as exc:
            return ActionResult(intent, "unknown", "Please check the tablet to confirm your message before trying again.",
                                {"diagnostic": exc.diagnostic})
        except (MessagingUnavailable, ValueError) as exc:
            return ActionResult(intent, "unavailable", "Messaging is unavailable. Please check the tablet connection and message access.",
                                {"diagnostic": getattr(exc, "diagnostic", {})})
        return ActionResult(intent, "submitted", "Your message has been sent successfully.", data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true", help="Show predefined text without Bluetooth or sending")
    mode.add_argument("--check", action="store_true", help="Connect to MAP and check SMS support; do not send")
    mode.add_argument("--send", action="store_true", help="Request a real carrier SMS")
    parser.add_argument("--speak", action="store_true")
    parser.add_argument("--voice-model", default=str(runtime_root() / "assets/voices/en_US-amy-medium.onnx"))
    args = parser.parse_args()
    from hey_swift.integration.runtime import prepare_workspace
    prepare_workspace()
    communications = communication_values()
    if args.preview:
        result = ActionResult("MESSAGE", "preview", DEMO_MESSAGE)
    else:
        try:
            backend = BluetoothMessagingBackend(communications["phone_mac"])
            if args.check:
                result = ActionResult("MESSAGE", "ready", "Bluetooth SMS interface is available. Sending still needs testing.", backend.check())
            else:
                result = MessagingService(backend, communications["message_number"]).execute()
        except (ValueError, MessagingUnavailable) as exc:
            result = ActionResult("MESSAGE", "unavailable", str(exc),
                                  {"diagnostic": getattr(exc, "diagnostic", {})})
    payload = result.to_dict()
    if args.speak:
        from hey_swift.audio.speech import PiperSpeech, SpeechService, SpeechUnavailable
        try:
            SpeechService(PiperSpeech(args.voice_model)).speak(result.message)
            payload["speech"] = {"status": "ok"}
        except SpeechUnavailable as exc:
            payload["speech"] = {"status": "unavailable", "message": str(exc)}
    print(json.dumps(payload, indent=2))
    if payload.get("speech", {}).get("status") == "unavailable":
        return 2
    return 0 if result.status in {"preview", "ready", "submitted"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
