"""kjob example: report the GPU, print progress for a few seconds, exit with a chosen code."""
import argparse
import os
import socket
import time

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--exit-code", type=int, default=0)
a = ap.parse_args()
print(f"host {socket.gethostname()}, node {os.environ.get('KJOB_NODE')}, cwd {os.getcwd()}", flush=True)
try:
    import torch
    print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-", flush=True)
except ImportError:
    print("torch not installed in this interpreter", flush=True)
for i in range(a.steps):
    print(f"step {i + 1}/{a.steps}", flush=True)
    time.sleep(2)
raise SystemExit(a.exit_code)
