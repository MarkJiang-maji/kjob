"""kjob command line: slurm-like commands for PyTorchJobs that mirror the current notebook.

  kjob submit  [opts] SCRIPT [ARGS...]   like sbatch: submit and return (job deleted by TTL after it finishes)
  kjob run     [opts] SCRIPT [ARGS...]   like srun:   submit, stream output live, delete the job, exit with its code
  kjob wait    NAME                       follow a submitted job until it ends, then delete it
  kjob queue   [--all] [-w]               like squeue: list jobs (use -w to refresh every few seconds)
  kjob logs    NAME [-f]                  print (or follow) a job's output; falls back to the saved log file
  kjob cancel  NAME... | --finished | --all-mine   like scancel
  kjob clean                              delete finished kjob jobs (same as cancel --finished)
  kjob info                               like sinfo: what jobs inherit from the notebook, quota usage, GPU types
Short aliases in bin/: ksub, ksrun, kwait, kqueue, klogs, kcancel, kclean, kinfo.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

from kubernetes.client.rest import ApiException

from . import core


def _add_job_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("-J", "--job-name", help="job name (default: script name + time)")
    p.add_argument("-G", "--gpus", type=int, default=1, help="GPUs (default 1; 0 for CPU-only)")
    p.add_argument("-c", "--cpus", type=int, default=4, help="CPU cores (default 4)")
    p.add_argument("--mem", default="32Gi", help="memory (default 32Gi)")
    p.add_argument("--shm", default="16Gi", help="/dev/shm size (default 16Gi)")
    p.add_argument("-g", "--gpu-type", default="", help=f"comma list of GPU types: {', '.join(core.GPU_PRODUCTS)} or a raw nvidia.com/gpu.product")
    p.add_argument("-w", "--nodelist", default=None, help="pin to this node (kubernetes.io/hostname)")
    p.add_argument("-x", "--exclude", default="", help="comma list of nodes to avoid")
    p.add_argument("--python", default=None, help="interpreter for the script (default: this Python for .py)")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="extra environment variable (repeatable)")
    p.add_argument("-D", "--chdir", default=None, help="working directory in the job (default: current directory)")
    p.add_argument("-o", "--log-dir", default=str(Path.home() / "kjob-logs"), help="where the job's output is saved")
    p.add_argument("--ttl", type=int, default=3600, help="delete a finished job after this many seconds if unmonitored")
    p.add_argument("script", help="script to run (.py with Python, .sh with bash, anything else executed directly)")
    p.add_argument("args", nargs=argparse.REMAINDER, help="arguments passed to the script")


def _spec(a) -> core.JobSpec:
    name = a.job_name or f"{Path(a.script).stem}-{dt.datetime.now().strftime('%m%d-%H%M%S')}"
    env = dict(e.split("=", 1) for e in a.env)
    return core.JobSpec(name=name, script=a.script, args=a.args, gpus=a.gpus, cpus=a.cpus, memory=a.mem, shm=a.shm,
                        gpu_types=[g for g in a.gpu_type.split(",") if g], node=a.nodelist,
                        exclude_nodes=[n for n in a.exclude.split(",") if n], interpreter=a.python, env=env,
                        workdir=a.chdir, log_dir=a.log_dir, ttl_seconds=a.ttl)


def _age(ts: str | None) -> str:
    if not ts:
        return "-"
    s = int((dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))).total_seconds())
    return f"{s // 86400}d{s % 86400 // 3600}h" if s >= 86400 else f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def _job_for(name: str) -> core.KJob:
    """KJob handle for an existing job (only the name is needed for monitoring)."""
    j = core.KJob.__new__(core.KJob)
    core._load_config()
    j.spec = core.JobSpec(name=name, script="")
    j.ns = core.current_namespace()
    j.core, j.custom = core.client.CoreV1Api(), core.client.CustomObjectsApi()
    j.pod_name = f"{name}-master-0"
    return j


def cmd_submit(a) -> int:
    job = core.KJob(_spec(a))
    name = job.submit()
    print(f"Submitted job {name}  (log: {job.spec.log_path()})")
    return 0


def cmd_run(a) -> int:
    return core.KJob(_spec(a)).run()


def cmd_wait(a) -> int:
    j = _job_for(a.name)
    report = lambda m: print(m, file=sys.stderr, flush=True)
    try:
        return j.follow(report=report)
    except KeyboardInterrupt:
        report("kjob: stopped following (job keeps running)")
        return 130
    finally:
        if not a.keep and j.state() in core.TERMINAL:
            j.delete()


def _queue_rows(everyone: bool):
    pods = {p.metadata.name: p for p in core.client.CoreV1Api().list_namespaced_pod(core.current_namespace()).items}
    rows = []
    for o in core.list_jobs(everyone=everyone):
        name = o["metadata"]["name"]
        pod = pods.get(f"{name}-master-0")
        tmpl = o["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"]["containers"][0]
        res = (tmpl.get("resources") or {}).get("requests", {})
        rows.append([name, core.job_state(o), pod.status.phase if pod else "-", pod.spec.node_name if pod and pod.spec.node_name else "-",
                     res.get("nvidia.com/gpu", "0"), res.get("cpu", "-"), _age(o["metadata"].get("creationTimestamp")),
                     (o["metadata"].get("annotations") or {}).get("kjob/command", "")[-60:]])
    return rows


def cmd_queue(a) -> int:
    while True:
        rows = _queue_rows(a.all)
        hdr = ["NAME", "STATE", "POD", "NODE", "GPU", "CPU", "AGE", "COMMAND"]
        w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
        if a.watch:
            print("\033[2J\033[H", end="")
        print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
        for r in rows:
            print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
        if not a.watch:
            return 0
        time.sleep(5)


def cmd_logs(a) -> int:
    j = _job_for(a.name)
    try:
        if a.follow:
            for line in j.stream_logs():
                print(line, flush=True)
        else:
            print(j.core.read_namespaced_pod_log(j.pod_name, j.ns), end="")
        return 0
    except (ApiException, KeyboardInterrupt):
        pass
    try:
        obj = j.custom.get_namespaced_custom_object(core.GROUP, core.VERSION, j.ns, core.PLURAL, a.name)
        log = (obj["metadata"].get("annotations") or {}).get("kjob/log")
    except ApiException:
        log = None
    log = log or str(Path(a.log_dir) / f"{a.name}.log")
    if os.path.exists(log):
        print(f"kjob: pod not available; showing {log}", file=sys.stderr)
        print(open(log).read(), end="")
        return 0
    print(f"kjob: no pod and no log file for {a.name}", file=sys.stderr)
    return 1


def cmd_cancel(a) -> int:
    ns = core.current_namespace()
    names = list(a.names)
    if a.finished or a.all_mine:
        names += [o["metadata"]["name"] for o in core.list_jobs() if a.all_mine or core.job_state(o) in core.TERMINAL]
    for n in names:
        print(f"{'deleted' if core.delete_job(ns, n) else 'not found'}: {n}")
    return 0


def cmd_info(a) -> int:
    t = core.NotebookTemplate.from_current_pod()
    print(f"notebook  {t.pod_name}  (namespace {t.namespace}, node {t.node})")
    print(f"image     {t.image}")
    print(f"workdir   {t.working_dir}  (jobs start here unless -D is given)")
    print("volumes inherited by jobs:")
    for v, m in zip(t.volumes, t.volume_mounts):
        kind = f"nfs {v.nfs.server}:{v.nfs.path}" if v.nfs else ("emptyDir" + (" (memory)" if v.empty_dir and v.empty_dir.medium == "Memory" else "")) if v.empty_dir else \
            f"pvc {v.persistent_volume_claim.claim_name}" if v.persistent_volume_claim else "other"
        print(f"  {m.mount_path:24s} {kind}")
    for n, why in t.skipped.items():
        print(f"  (not copied: {n} - {why})")
    print("quota (used / hard):")
    for k, used, hard in core.quota():
        print(f"  {k:28s} {used} / {hard}")
    print("GPU types for -g:", ", ".join(f"{k}={v}" for k, v in core.GPU_PRODUCTS.items()))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kjob", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, help_ in [("submit", cmd_submit, "submit and return (sbatch)"), ("run", cmd_run, "submit, stream output, clean up (srun)")]:
        p = sub.add_parser(name, help=help_)
        _add_job_opts(p)
        p.set_defaults(fn=fn)
    p = sub.add_parser("wait", help="follow a job until it ends, then delete it")
    p.add_argument("name")
    p.add_argument("--keep", action="store_true", help="do not delete the job when it ends")
    p.set_defaults(fn=cmd_wait)
    p = sub.add_parser("queue", help="list jobs (squeue)")
    p.add_argument("--all", action="store_true", help="include PyTorchJobs not created by kjob")
    p.add_argument("-w", "--watch", action="store_true", help="refresh every 5 s")
    p.set_defaults(fn=cmd_queue)
    p = sub.add_parser("logs", help="print or follow a job's output")
    p.add_argument("name")
    p.add_argument("-f", "--follow", action="store_true")
    p.add_argument("-o", "--log-dir", default=str(Path.home() / "kjob-logs"))
    p.set_defaults(fn=cmd_logs)
    p = sub.add_parser("cancel", help="delete jobs (scancel)")
    p.add_argument("names", nargs="*")
    p.add_argument("--finished", action="store_true", help="delete all finished kjob jobs")
    p.add_argument("--all-mine", action="store_true", help="delete every kjob job")
    p.set_defaults(fn=cmd_cancel)
    p = sub.add_parser("clean", help="delete finished kjob jobs")
    p.set_defaults(fn=lambda a: cmd_cancel(argparse.Namespace(names=[], finished=True, all_mine=False)))
    p = sub.add_parser("info", help="show inherited environment, quota and GPU types (sinfo)")
    p.set_defaults(fn=cmd_info)
    a = ap.parse_args(argv)
    core._load_config()
    if getattr(a, "args", None) and a.args[:1] == ["--"]:
        a.args = a.args[1:]
    return a.fn(a)
