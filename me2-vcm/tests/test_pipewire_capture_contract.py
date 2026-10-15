"""The shared capture subprocess must emit mono float32 PCM, not a container."""
import io
from unittest.mock import Mock, patch

from hey_swift.audio.pipewire_capture import PipeWireInputStream


def test_explicit_source_channel_map_and_raw_pcm():
    process = Mock()
    process.stdout = io.BytesIO(b"")
    process.poll.return_value = 0
    with patch("hey_swift.audio.pipewire_capture.subprocess.Popen", return_value=process) as spawn:
        with PipeWireInputStream("verified.usb.source", samplerate=48000, blocksize=4800,
                                 dtype="float32", channels=1):
            pass
    command = spawn.call_args.args[0]
    assert command[command.index("--target") + 1] == "verified.usb.source"
    assert command[command.index("--channel-map") + 1] == "MONO"
    assert "--channelmap" not in command
    assert "--raw" in command
    assert command[command.index("--format") + 1] == "f32"
    assert command[-1] == "-"
