import os, json, subprocess, sys

ONIT_PY = "/home/edgardo.callagon.jr/.conda/envs/onit/bin/python"
HOME = os.path.expanduser("~")
ks_dir = os.path.join(HOME, ".local", "share", "jupyter", "kernels", "onit312")
os.makedirs(ks_dir, exist_ok=True)

kernel_json = {
    "argv": [ONIT_PY, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "OnIt Python 3.12",
    "language": "python",
}
with open(os.path.join(ks_dir, "kernel.json"), "w") as f:
    json.dump(kernel_json, f, indent=2)
print("Wrote kernel spec to", ks_dir)

# Verify ipykernel is importable in onit env
r = subprocess.run([ONIT_PY, "-c", "import ipykernel; print('ipykernel', ipykernel.__version__)"],
                   capture_output=True, text=True)
print("ipykernel check:", r.stdout.strip() or r.stderr.strip())
