"""Low-rate Linux worker resource sampling; no privileged controls."""
import json
import os
from pathlib import Path
import threading
import time


class ResourceSampler:
    def __init__(self, pid, output):
        self.pid, self.output = pid, output
        self.rows = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def run(self):
        previous = None
        while not self.stop.is_set():
            try:
                status = {}
                for line in Path(f"/proc/{self.pid}/status").read_text().splitlines():
                    name, _, value = line.partition(":")
                    if name in ("VmRSS", "Threads"):
                        status[name] = int(value.strip().split()[0])
                fields = Path(f"/proc/{self.pid}/stat").read_text().rsplit(")", 1)[1].split()
                cpu = (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
                now = time.monotonic()
                row = dict(monotonic_seconds=now, rss_kib=status.get("VmRSS"), threads=status.get("Threads"),
                           cpu_percent_one_core=(100*(cpu-previous[1])/(now-previous[0]) if previous else None))
                previous = (now, cpu)
                for path, key, divisor in (("/sys/class/thermal/thermal_zone0/temp", "temperature_c", 1000),
                                          ("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "cpu0_frequency_mhz", 1000)):
                    try:
                        row[key] = float(Path(path).read_text()) / divisor
                    except (OSError, ValueError):
                        row[key] = None
                self.rows.append(row)
            except (OSError, ValueError, IndexError):
                pass
            self.stop.wait(.5)

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=2)
        summary = dict(scope="evaluation_worker_process_only; sampled 0.5s; includes fetch/decode/evaluation; not per-inference peaks",
                       samples=self.rows, n=len(self.rows))
        for key in ("rss_kib", "cpu_percent_one_core", "temperature_c", "cpu0_frequency_mhz"):
            values = [r[key] for r in self.rows if r.get(key) is not None]
            summary[key] = dict(max=max(values) if values else None,
                                mean=sum(values)/len(values) if values else None, n=len(values))
        self.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
