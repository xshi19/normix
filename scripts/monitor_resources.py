"""
Background resource monitor — samples CPU, RAM, and GPU every INTERVAL seconds.
Writes a CSV to the given output path, then prints a summary.

Usage:
    python scripts/monitor_resources.py --pid PID --output out.csv [--interval 2]
"""

import argparse
import csv
import subprocess
import time
import datetime
import psutil

INTERVAL = 2  # seconds between samples


def _gpu_stats():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=3,
        ).decode().strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "gpu_util_pct": float(parts[0]),
            "gpu_mem_util_pct": float(parts[1]),
            "gpu_mem_used_mb": float(parts[2]),
            "gpu_mem_total_mb": float(parts[3]),
        }
    except Exception:
        return {
            "gpu_util_pct": None,
            "gpu_mem_util_pct": None,
            "gpu_mem_used_mb": None,
            "gpu_mem_total_mb": None,
        }


def monitor(pid: int, output: str, interval: float = INTERVAL):
    proc = psutil.Process(pid)
    fieldnames = [
        "timestamp", "elapsed_s",
        "cpu_pct_proc", "cpu_pct_total",
        "mem_rss_mb", "mem_vms_mb", "mem_pct",
        "gpu_util_pct", "gpu_mem_util_pct",
        "gpu_mem_used_mb", "gpu_mem_total_mb",
    ]
    start = time.monotonic()

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while True:
            try:
                proc.status()  # raises NoSuchProcess when done
            except psutil.NoSuchProcess:
                break

            elapsed = time.monotonic() - start
            try:
                cpu_proc = proc.cpu_percent(interval=None)
                mem = proc.memory_info()
                mem_pct = proc.memory_percent()
                # include child processes
                children = proc.children(recursive=True)
                rss_total = mem.rss
                vms_total = mem.vms
                for child in children:
                    try:
                        cpu_proc += child.cpu_percent(interval=None)
                        child_mem = child.memory_info()
                        rss_total += child_mem.rss
                        vms_total += child_mem.vms
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                mem = type("mem", (), {"rss": rss_total, "vms": vms_total})()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

            gpu = _gpu_stats()
            row = {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "elapsed_s": round(elapsed, 1),
                "cpu_pct_proc": round(cpu_proc, 1),
                "cpu_pct_total": round(psutil.cpu_percent(interval=None), 1),
                "mem_rss_mb": round(mem.rss / 1024**2, 1),
                "mem_vms_mb": round(mem.vms / 1024**2, 1),
                "mem_pct": round(mem_pct, 2),
                **gpu,
            }
            writer.writerow(row)
            f.flush()
            time.sleep(interval)

    # print peak summary
    rows = []
    with open(output) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No samples collected.")
        return

    def _peak(key):
        vals = [float(r[key]) for r in rows if r[key] not in (None, "", "None")]
        return max(vals) if vals else None

    print("\n=== Resource Monitor Summary ===")
    print(f"  Samples collected : {len(rows)}")
    print(f"  Total elapsed     : {rows[-1]['elapsed_s']} s")
    print(f"  Peak CPU (proc)   : {_peak('cpu_pct_proc')} %")
    print(f"  Peak CPU (total)  : {_peak('cpu_pct_total')} %")
    print(f"  Peak RAM (RSS)    : {_peak('mem_rss_mb')} MB")
    print(f"  Peak GPU util     : {_peak('gpu_util_pct')} %")
    print(f"  Peak GPU mem      : {_peak('gpu_mem_used_mb')} MB")
    print(f"  CSV saved to      : {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=INTERVAL)
    args = parser.parse_args()
    monitor(args.pid, args.output, args.interval)
