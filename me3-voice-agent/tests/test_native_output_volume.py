import asyncio

from me3_voice import native_bridge


def test_wpctl_volume_parser_accepts_only_normalized_levels():
    assert native_bridge._parse_wpctl_volume("Volume: 0.60") == 0.6
    assert native_bridge._parse_wpctl_volume("Volume: 1.00 [MUTED]") == 1.0
    assert native_bridge._parse_wpctl_volume("Volume: 1.20") is None
    assert native_bridge._parse_wpctl_volume("no sink") is None


def test_pipewire_volume_uses_only_default_sink_and_reads_back(monkeypatch):
    calls = []

    class Process:
        def __init__(self, output=b""):
            self.output = output
            self.returncode = 0

        async def communicate(self):
            return self.output, b""

    async def create(*args, **kwargs):
        calls.append((args, kwargs))
        return Process(b"" if args[1] == "set-volume" else b"Volume: 0.60\n")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    assert asyncio.run(native_bridge._pipewire_output_volume(0.6)) == 0.6
    assert [call[0] for call in calls] == [
        ("wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "0.60"),
        ("wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"),
    ]


def test_dashboard_has_compact_bridge_speaker_volume_control():
    from pathlib import Path

    html = (Path(__file__).parents[1] / "src/me3_voice/ui/dashboard.html").read_text()
    script = (Path(__file__).parents[1] / "src/me3_voice/ui/dashboard.js").read_text()
    assert 'id="output-volume" type="range" min="0" max="1"' in html
    assert "Adjusts the Swift bridge machine's default speaker output" in html
    assert "fetch('/api/audio/output-volume'" in script
    assert "method:'POST'" in script
    assert "value.target_matched!==true" in script


def test_native_dashboard_hides_playback_from_non_target_spotify_device():
    import shutil
    import subprocess
    from pathlib import Path

    import pytest

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to execute the browser music-state helper")
    source = (Path(__file__).parents[1] / "src/me3_voice/ui/dashboard.js").read_text()
    start = source.index("  function setMusicState(")
    end = source.index("  function renderMusic(", start)
    helper = source[start:end]
    script = (
        "const runtime={music:{capabilities:{output_backend:'spotify_connect',target_configured:true,device_name:'Swift Soloist'}}};"
        "const reports=[];const numeric=Number.isFinite;const safeArtwork=()=>'';"
        "const performance={now:()=>10};const renderMusic=()=>{};"
        "const reportMusicPlayback=(v,s)=>reports.push([v,s]);\n"
        + helper
        + "setMusicState({status:'playing',device_name:'Alexa',target_matched:false,track:{name:'Private other track',artists:'Other'},volume_percent:40},'api');"
        + "if(runtime.music.state.track!==null||runtime.music.state.device_name!==null||reports.length)throw new Error('Non-target playback leaked into Swift UI');"
        + "setMusicState({status:'playing',device_name:'Swift Soloist',target_matched:true,track:{name:'Swift track',artists:'Swift artist'}},'api');"
        + "if(runtime.music.state.track?.name!=='Swift track')throw new Error('Swift Soloist playback was hidden');"
    )
    subprocess.run([node, "-e", script], capture_output=True, text=True, check=True, timeout=5)
