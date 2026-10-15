"""kjob example: show where the job runs, use the GPU for a few steps, exit with a chosen code."""
import argparse
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--exit-code", type=int, default=0)
a = ap.parse_args()
print(f"job {os.environ.get('KJOB_NAME')} on node {os.environ.get('KJOB_NODE')}, cwd {os.getcwd()}", flush=True)
try:
    import torch
except ImportError:
    torch = None
    print("torch is not installed in this Python; only printing steps", flush=True)
cuda = torch is not None and torch.cuda.is_available()
if cuda:
    print(f"GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}", flush=True)
    x = torch.randn(4096, 4096, device="cuda")
elif torch is not None:
    print("CUDA is not available (CPU-only job, -G 0?)", flush=True)
for i in range(a.steps):
    if cuda:
        (x @ x).sum().item()           # a little real GPU work
    print(f"step {i + 1}/{a.steps} ok", flush=True)
    time.sleep(1)
raise SystemExit(a.exit_code)
