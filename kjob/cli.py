"""kjob command line: slurm-like commands for PyTorchJobs that mirror the current notebook.

  kjob submit  [opts] SCRIPT [ARGS...]   like sbatch: submit and return (job deleted by TTL after it finishes)
  kjob run     [opts] SCRIPT [ARGS...]   like srun:   submit, stream output live, delete the job, exit with its code
  kjob wait    NAME                       follow a submitted job until it ends, then delete it
  kjob queue   [--all] [-w]               like squeue: list jobs (use -w to refresh every few seconds)
  kjob logs    NAME [-f]                  print (or follow) a job's output; falls back to the saved log file
  kjob cancel  NAME... | --finished | --all-mine [--all-notebooks]   like scancel
  kjob clean   [--all-notebooks]          delete finished kjob jobs (same as cancel --finished)
  kjob info                               like sinfo: what jobs inherit from the notebook, quota usage, GPU types
Short aliases (installed by pip, also in bin/): ksub, ksrun, kwait, kqueue, klogs, kcancel, kclean, kinfo.
--finished, --all-mine and clean only touch jobs submitted from this notebook unless --all-notebooks is given.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

from kubernetes.client.rest import ApiException
from urllib3.exceptions import HTTPError

from . import core


def _add_job_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("-J", "--job-name", help="job name (default: script name + time)")
    p.add_argument("-G", "--gpus", type=int, default=1, help="GPUs (default 1; 0 for CPU-only)")
    p.add_argument("-c", "--cpus", type=int, default=core.DEFAULT_CPUS, help="CPU cores (default 1)")
    p.add_argument("--mem", default=core.DEFAULT_MEMORY, help="memory (default 2Gi)")
    p.add_argument("--shm", default=core.DEFAULT_SHM, help="/dev/shm size (default 256Mi)")
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
    name = a.job_name or core.unique_name(Path(a.script).stem)
    if any("=" not in e for e in a.env):
        raise ValueError("--env must be KEY=VALUE.")
    env = dict(e.split("=", 1) for e in a.env)
    spec = core.JobSpec(name=name, script=a.script, args=a.args, gpus=a.gpus, cpus=a.cpus, memory=a.mem, shm=a.shm,
                        gpu_types=[g for g in a.gpu_type.split(",") if g], node=a.nodelist,
                        exclude_nodes=[n for n in a.exclude.split(",") if n], interpreter=a.python, env=env,
                        workdir=str(Path(a.chdir or os.getcwd()).resolve()),
                        log_dir=str(Path(a.log_dir).expanduser().resolve()), ttl_seconds=a.ttl)
    spec.validate()
    return spec


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
        rows.append([name, core.job_state(o), (o["metadata"].get("labels") or {}).get("kjob/notebook", "-"),
                     pod.status.phase if pod else "-", pod.spec.node_name if pod and pod.spec.node_name else "-",
                     res.get("nvidia.com/gpu", "0"), res.get("cpu", "-"), _age(o["metadata"].get("creationTimestamp")),
                     (o["metadata"].get("annotations") or {}).get("kjob/command", "")[-60:]])
    return rows


def cmd_queue(a) -> int:
    while True:
        rows = _queue_rows(a.all)
        hdr = ["NAME", "STATE", "NOTEBOOK", "POD", "NODE", "GPU", "CPU", "AGE", "COMMAND"]
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
    state = j.state()
    try:
        if state == "Deleted":
            raise ApiException(status=404)
        if a.follow:
            for line in j.stream_logs():
                print(line, flush=True)
        else:
            j._pod()
            print(j.core.read_namespaced_pod_log(j.pod_name, j.ns), end="")
        return 0
    except KeyboardInterrupt:
        return 130
    except ApiException as e:
        if e.status != 404:
            raise
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
    targets = {n: None for n in a.names}
    if a.finished or a.all_mine:
        notebook = None if a.all_notebooks else core.current_notebook()
        for o in core.list_jobs(notebook=notebook):
            if a.all_mine or core.job_state(o) in core.TERMINAL:
                targets[o["metadata"]["name"]] = o["metadata"].get("uid")
    for n, uid in targets.items():
        deleted = core.delete_job(ns, n, uid=uid)
        print(f"{'deleted' if deleted else 'not found or replaced'}: {n}")
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
    p.add_argument("--all-mine", action="store_true", help="delete every kjob job submitted from this notebook")
    p.add_argument("--all-notebooks", action="store_true", help="--finished / --all-mine: include jobs from other notebooks")
    p.set_defaults(fn=cmd_cancel)
    p = sub.add_parser("clean", help="delete finished kjob jobs submitted from this notebook")
    p.add_argument("--all-notebooks", action="store_true", help="include jobs from other notebooks")
    p.set_defaults(fn=lambda a: cmd_cancel(argparse.Namespace(names=[], finished=True, all_mine=False,
                                                              all_notebooks=a.all_notebooks)))
    p = sub.add_parser("info", help="show inherited environment, quota and GPU types (sinfo)")
    p.set_defaults(fn=cmd_info)
    a = ap.parse_args(argv)
    core._load_config()
    if getattr(a, "args", None) and a.args[:1] == ["--"]:
        a.args = a.args[1:]
    try:
        return a.fn(a)
    except ValueError as e:
        ap.error(str(e))
    except ApiException as e:
        ap.exit(1, f"kjob: Kubernetes API error ({e.status}): {e.reason}\n")
    except HTTPError as e:  # connection refused/reset, timeouts: the job itself is unaffected
        ap.exit(1, f"kjob: lost connection to the Kubernetes API ({type(e).__name__}); submitted jobs keep running, "
                   f"check them later with kqueue / kwait NAME\n")


# console scripts installed by pip (pyproject.toml); the bin/ wrappers do the same for a source checkout
def ksub() -> int: return main(["submit", *sys.argv[1:]])
def ksrun() -> int: return main(["run", *sys.argv[1:]])
def kwait() -> int: return main(["wait", *sys.argv[1:]])
def kqueue() -> int: return main(["queue", *sys.argv[1:]])
def klogs() -> int: return main(["logs", *sys.argv[1:]])
def kcancel() -> int: return main(["cancel", *sys.argv[1:]])
def kclean() -> int: return main(["clean", *sys.argv[1:]])
def kinfo() -> int: return main(["info", *sys.argv[1:]])
