"""Optional same-terminal music controls polled between microphone reads."""
import select
import sys

from hey_swift.actions.services.music import MusicUnavailable


HELP = ":volume 10 | :quieter | :pause | :stop | :help (type then Enter)"


class TerminalMusicControls:
    def __init__(self, router, *, emit, stream=None):
        self.router = router
        self.emit = emit
        self.stream = sys.stdin if stream is None else stream

    def poll(self):
        if not self.stream.isatty():
            return False
        try:
            ready, _, _ = select.select([self.stream], [], [], 0)
            if not ready:
                return False
            line = self.stream.readline()
        except (OSError, ValueError):
            return False
        if not line:
            return False
        self.handle(line.strip())
        return True

    def handle(self, line):
        if not line:
            return
        if line == ":help":
            self.emit({"event": "terminal_control", "message": HELP})
            return
        service = self.router.services.get("music")
        if service is None:
            self.emit({"event": "terminal_control", "status": "unavailable",
                       "message": "Music is not enabled in this checkout's setup."})
            return
        try:
            if line.startswith(":volume "):
                value = line.split(maxsplit=1)[1]
                if not value.isascii() or not value.isdecimal():
                    raise MusicUnavailable("Use :volume followed by a whole percentage, e.g. :volume 10.")
                data = service.backend.set_volume(int(value))
                event = {"status": "ok", "message": "Music volume updated.", **data}
            elif line in {":quieter", ":pause", ":stop"}:
                intent = {":quieter": "VOLUME_DOWN", ":pause": "PAUSE", ":stop": "STOP"}[line]
                result = service.execute(intent)
                event = {"status": result.status, "message": result.message, **(result.data or {})}
            else:
                raise MusicUnavailable("Unknown terminal command. " + HELP)
        except MusicUnavailable as exc:
            self.emit({"event": "terminal_control", "status": "unavailable", "message": str(exc)})
            return
        except (OSError, ValueError):
            self.emit({"event": "terminal_control", "status": "unavailable",
                       "message": "Music control failed; check the selected player. " + HELP})
            return
        self.emit({"event": "terminal_control", **event})
