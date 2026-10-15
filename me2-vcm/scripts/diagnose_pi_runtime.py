#!/usr/bin/env python3
"""Read-only Pi audio evidence; no profile changes, playback, calls or SMS."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from hey_swift.integration.config import project_root, runtime_root, spotify_paths
from hey_swift.integration.e2e_loop import PATCH_VERSION


def command(argv, max_output=20000):
    try:
        result=subprocess.run(argv,capture_output=True,text=True,timeout=5)
        return {"returncode":result.returncode,"stdout":result.stdout if max_output is None else result.stdout[-max_output:],"stderr":result.stderr[-2000:]}
    except (OSError,subprocess.SubprocessError) as exc:
        return {"unavailable":type(exc).__name__}


def main():
    report={"python":sys.executable,"version":sys.version,"runtime":PATCH_VERSION}
    try:
        import sounddevice as sd
        report["sounddevice"]={"devices":list(sd.query_devices()),"hostapis":list(sd.query_hostapis())}
    except Exception as exc:
        report["sounddevice"]={"unavailable":type(exc).__name__,"message":str(exc)}
    for label,argv in {
        "capture_hardware":["arecord","-l"], "pipewire":["wpctl","status","-n"],
        "sink_volume":["wpctl","get-volume","@DEFAULT_AUDIO_SINK@"],
        "source_volume":["wpctl","get-volume","@DEFAULT_AUDIO_SOURCE@"],
        "usb":["lsusb"], "alsa_capture_users":["fuser","-v","/dev/snd/pcmC2D0c"],
    }.items(): report[label]=command(argv)
    graph=command(["pw-dump"], max_output=None)
    try:
        fields={"node.name","media.class","device.name","audio.rate","audio.channels","api.alsa.path",
                "target.object","object.serial","media.role","bluez5.profile","bluez5.codec"}
        report["pipewire_nodes"]=[{"id":item.get("id"),"type":item.get("type"),
            "state":item.get("info",{}).get("state"),
            "input_node":item.get("info",{}).get("input-node-id"),
            "output_node":item.get("info",{}).get("output-node-id"),
            "props":{k:v for k,v in item.get("info",{}).get("props",{}).items() if k in fields}}
            for item in json.loads(graph.get("stdout","")) if item.get("type","").endswith(("Node","Device","Link"))]
        report["bluetooth_stream_volumes"]={str(node["id"]):command(["wpctl","get-volume",str(node["id"])])
            for node in report["pipewire_nodes"] if node["props"].get("node.name","").startswith(("bluez_input.","bluez_output."))}
    except (ValueError,TypeError): report["pipewire_nodes"]={"unavailable":True}
    cue=project_root()/"audio/wake_chime.wav"
    if cue.is_file():
        from hey_swift.audio.cue import prepared_cue
        with prepared_cue(cue) as (_,levels): report["cue"]=levels
    selected=spotify_paths(runtime_root())
    # Never print account state, API keys, command trace, track metadata or recipients.
    report["spotify_profile"]=selected["profile"]
    state=command([str(selected["binary"]),"ctl","now","--json","-D",str(selected["data_dir"])])
    try:
        values=json.loads(state.get("stdout",""))
        report["spotify"]={key:values.get(key) for key in ("status","is_active","volume")}
    except (ValueError,TypeError): report["spotify"]={"unavailable":True,"returncode":state.get("returncode")}
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
