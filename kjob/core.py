"""Submit, monitor and clean up Kubeflow PyTorchJobs that mirror the current notebook's environment.

The job copies the notebook pod's image, working directory, security context and the volumes mounted into the
notebook container, so code, virtualenvs and data on those volumes are visible at the same paths. Volumes that a
namespace PodDefault injects are left out (the PodDefault webhook adds them again and rejects duplicates).
ReadWriteOnce claims that are backed by NFS are mounted as the underlying NFS export so jobs can run on other nodes.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
import shlex
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException

GROUP, VERSION, PLURAL = "kubeflow.org", "v1", "pytorchjobs"
MANAGED_BY = "kjob"
SIDECAR_CONTAINERS = {"istio-proxy"}
GPU_PRODUCTS = {  # short names for --gpu-type; any other value is used as the raw nvidia.com/gpu.product label
    "a5000": "NVIDIA-RTX-A5000",
    "v100": "Tesla-V100-SXM2-16GB",
}
TERMINAL = ("Succeeded", "Failed")


def _load_config() -> None:
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()


def current_namespace() -> str:
    f = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
    return f.read_text().strip() if f.exists() else os.environ.get("NB_NAMESPACE", "default")


def sanitize_name(name: str) -> str:
    """Kubernetes object names: lowercase alphanumerics and '-', at most 50 chars (pods add '-master-0')."""
    n = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    if not n:
        raise ValueError(f"invalid job name {name!r}")
    return n[:50].rstrip("-")


def _nfs_mounts() -> dict[str, tuple[str, str]]:
    """mount point -> (server, export path) for NFS mounts visible in this container."""
    out = {}
    for line in Path("/proc/mounts").read_text().splitlines():
        src, mnt, fstype = line.split()[:3]
        if fstype.startswith("nfs") and ":" in src:
            server, path = src.split(":", 1)
            out[mnt] = (server, path)
    return out


@dataclass
class NotebookTemplate:
    """What a job inherits from the notebook pod this code runs in."""
    namespace: str
    pod_name: str
    node: str
    image: str
    working_dir: str
    volumes: list = field(default_factory=list)          # V1Volume
    volume_mounts: list = field(default_factory=list)    # V1VolumeMount
    security_context: object = None
    image_pull_secrets: list = field(default_factory=list)
    skipped: dict = field(default_factory=dict)          # volume name -> reason

    @classmethod
    def from_current_pod(cls, shm_size: str = "16Gi") -> "NotebookTemplate":
        _load_config()
        ns = current_namespace()
        core = client.CoreV1Api()
        pod = core.read_namespaced_pod(socket.gethostname(), ns)
        main = [c for c in pod.spec.containers if c.name not in SIDECAR_CONTAINERS][0]
        injected = set()
        try:
            for pd in client.CustomObjectsApi().list_namespaced_custom_object("kubeflow.org", "v1alpha1", ns, "poddefaults")["items"]:
                injected |= {v["name"] for v in pd["spec"].get("volumes", [])}
        except ApiException:
            pass
        mounts = {m.name: m for m in main.volume_mounts or []}
        nfs = _nfs_mounts()
        t = cls(namespace=ns, pod_name=pod.metadata.name, node=pod.spec.node_name, image=main.image,
                working_dir=os.getcwd(), security_context=pod.spec.security_context,
                image_pull_secrets=pod.spec.image_pull_secrets or [])
        for v in pod.spec.volumes or []:
            m = mounts.get(v.name)
            if m is None:
                t.skipped[v.name] = "not mounted in the notebook container"
            elif v.name in injected:
                t.skipped[v.name] = "added automatically by a PodDefault"
            elif v.projected is not None or v.name.startswith("kube-api-access"):
                t.skipped[v.name] = "service-account token"
            elif v.persistent_volume_claim is not None and m.mount_path in nfs:
                server, path = nfs[m.mount_path]
                t.volumes.append(client.V1Volume(name=v.name, nfs=client.V1NFSVolumeSource(server=server, path=path)))
                t.volume_mounts.append(client.V1VolumeMount(name=v.name, mount_path=m.mount_path, sub_path=m.sub_path))
            elif v.empty_dir is not None:
                medium = v.empty_dir.medium
                t.volumes.append(client.V1Volume(name=v.name, empty_dir=client.V1EmptyDirVolumeSource(
                    medium=medium, size_limit=shm_size if medium == "Memory" else None)))
                t.volume_mounts.append(client.V1VolumeMount(name=v.name, mount_path=m.mount_path))
            else:  # PVC on non-NFS storage, nfs, secret, configMap, ...: reuse as-is
                t.volumes.append(v)
                t.volume_mounts.append(client.V1VolumeMount(name=v.name, mount_path=m.mount_path,
                                                            sub_path=m.sub_path, read_only=m.read_only))
        return t

    def shared_paths(self) -> list[str]:
        """Mount points shared with jobs (emptyDir volumes are per pod)."""
        names = {v.name for v in self.volumes if v.empty_dir is None}
        return [m.mount_path for m in self.volume_mounts if m.name in names]


@dataclass
class JobSpec:
    name: str
    script: str
    args: list[str] = field(default_factory=list)
    gpus: int = 1
    cpus: int = 4
    memory: str = "32Gi"
    shm: str = "16Gi"
    gpu_types: list[str] = field(default_factory=list)   # e.g. ["a5000", "v100"]
    node: str | None = None                              # pin to one host
    exclude_nodes: list[str] = field(default_factory=list)
    interpreter: str | None = None                       # default: this Python for .py, bash for .sh
    env: dict[str, str] = field(default_factory=dict)
    workdir: str | None = None
    log_dir: str = str(Path.home() / "kjob-logs")
    ttl_seconds: int = 3600                              # safety net if nobody monitors the job

    def command(self) -> str:
        script = str(Path(self.script).resolve()) if Path(self.script).exists() else self.script
        if self.interpreter:
            prog = [self.interpreter, script]
        elif script.endswith(".py"):
            prog = [sys.executable, script]
        elif script.endswith(".sh"):
            prog = ["bash", script]
        else:
            prog = [script]
        return " ".join(shlex.quote(p) for p in prog + list(self.args))

    def log_path(self) -> str:
        return str(Path(self.log_dir) / f"{self.name}.log")


class KJob:
    """Create / monitor / delete one PyTorchJob built from a NotebookTemplate."""

    def __init__(self, spec: JobSpec, template: NotebookTemplate | None = None):
        self.spec = spec
        self.spec.name = sanitize_name(spec.name)
        self.t = template or NotebookTemplate.from_current_pod(spec.shm)
        self.ns = self.t.namespace
        self.core = client.CoreV1Api()
        self.custom = client.CustomObjectsApi()
        self.pod_name = f"{self.spec.name}-master-0"

    # ---------- manifest ----------
    def manifest(self) -> dict:
        s, t = self.spec, self.t
        workdir = s.workdir or t.working_dir
        log = s.log_path()
        shell = (f"set -o pipefail; mkdir -p {shlex.quote(str(Path(log).parent))}; cd {shlex.quote(workdir)} && "
                 f"{{ echo \"[kjob $(date -u +%FT%TZ)] start {s.name} on node $KJOB_NODE\"; {s.command()}; rc=$?; "
                 f"echo \"[kjob $(date -u +%FT%TZ)] exit code $rc\"; exit $rc; }} "
                 f"2>&1 | tee -a {shlex.quote(log)}")
        res = {"cpu": str(s.cpus), "memory": s.memory}
        if s.gpus:
            res["nvidia.com/gpu"] = str(s.gpus)  # CPU-only jobs must not request GPUs at all
        env = [client.V1EnvVar(name="PYTHONUNBUFFERED", value="1"),
               client.V1EnvVar(name="KJOB_NAME", value=s.name),
               client.V1EnvVar(name="KJOB_NODE", value_from=client.V1EnvVarSource(
                   field_ref=client.V1ObjectFieldSelector(field_path="spec.nodeName")))]
        env += [client.V1EnvVar(name=k, value=v) for k, v in s.env.items()]
        container = client.V1Container(
            name="pytorch", image=t.image, command=["bash", "-lc", shell], working_dir=workdir, env=env,
            resources=client.V1ResourceRequirements(requests=res, limits=res), volume_mounts=t.volume_mounts)
        terms = []
        if s.node:
            terms.append(client.V1NodeSelectorRequirement(key="kubernetes.io/hostname", operator="In", values=[s.node]))
        if s.gpu_types:
            terms.append(client.V1NodeSelectorRequirement(key="nvidia.com/gpu.product", operator="In",
                                                          values=[GPU_PRODUCTS.get(g, g) for g in s.gpu_types]))
        if s.exclude_nodes:
            terms.append(client.V1NodeSelectorRequirement(key="kubernetes.io/hostname", operator="NotIn", values=s.exclude_nodes))
        affinity = client.V1Affinity(node_affinity=client.V1NodeAffinity(
            required_during_scheduling_ignored_during_execution=client.V1NodeSelector(
                node_selector_terms=[client.V1NodeSelectorTerm(match_expressions=terms)]))) if terms else None
        pod_spec = client.V1PodSpec(containers=[container], volumes=t.volumes, affinity=affinity,
                                    security_context=t.security_context, image_pull_secrets=t.image_pull_secrets or None)
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(annotations={"sidecar.istio.io/inject": "false"},
                                         labels={"app.kubernetes.io/managed-by": MANAGED_BY}),
            spec=pod_spec)
        api = client.ApiClient()
        return {
            "apiVersion": f"{GROUP}/{VERSION}", "kind": "PyTorchJob",
            "metadata": {"name": s.name, "namespace": self.ns,
                         "labels": {"app.kubernetes.io/managed-by": MANAGED_BY, "kjob/notebook": t.pod_name},
                         "annotations": {"kjob/command": s.command()[:2000], "kjob/log": log,
                                         "kjob/workdir": workdir, "kjob/submitted": _dt.datetime.utcnow().isoformat() + "Z"}},
            "spec": {
                "runPolicy": {"cleanPodPolicy": "None", "backoffLimit": 0, "ttlSecondsAfterFinished": s.ttl_seconds},
                "pytorchReplicaSpecs": {"Master": {"replicas": 1, "restartPolicy": "Never",
                                                   "template": api.sanitize_for_serialization(template)}},
            },
        }

    # ---------- lifecycle ----------
    def submit(self) -> str:
        self._warn_unshared_paths()
        self.custom.create_namespaced_custom_object(GROUP, VERSION, self.ns, PLURAL, self.manifest())
        return self.spec.name

    def _warn_unshared_paths(self) -> None:
        shared = self.t.shared_paths()
        for p in [self.spec.workdir or self.t.working_dir, self.spec.log_dir, self.spec.script]:
            if p.startswith("/") and not any(p == s or p.startswith(s.rstrip("/") + "/") for s in shared):
                print(f"kjob: warning: {p} is not on a volume shared with jobs ({', '.join(shared)})", file=sys.stderr)

    def state(self) -> str:
        return job_state(self.custom.get_namespaced_custom_object(GROUP, VERSION, self.ns, PLURAL, self.spec.name))

    def delete(self) -> None:
        delete_job(self.ns, self.spec.name)

    def wait_for_pod(self, timeout: float | None = None, report=print) -> str:
        """Block until the pod leaves Pending (or the job ends); report scheduling problems while waiting."""
        t0, seen = time.time(), set()
        while True:
            st = self.state()
            try:
                pod = self.core.read_namespaced_pod(self.pod_name, self.ns)
                phase = pod.status.phase
                if phase != "Pending":
                    return phase
                for c in pod.status.conditions or []:
                    if c.type == "PodScheduled" and c.status == "False" and c.message and c.message not in seen:
                        seen.add(c.message)
                        report(f"kjob: {self.spec.name} pending: {c.message}")
                for cs in pod.status.container_statuses or []:
                    w = cs.state.waiting
                    if w and w.reason not in (None, "ContainerCreating") and w.reason not in seen:
                        seen.add(w.reason)
                        report(f"kjob: {self.spec.name} waiting: {w.reason} {w.message or ''}")
            except ApiException as e:
                if e.status != 404:
                    raise
                for ev in job_events(self.ns, self.spec.name):
                    if ev not in seen:
                        seen.add(ev)
                        report(f"kjob: {self.spec.name} event: {ev}")
            if st in TERMINAL:
                return st
            if timeout is not None and time.time() - t0 > timeout:
                return "Pending"
            time.sleep(3)

    def stream_logs(self) -> Iterator[str]:
        """Yield log lines in real time until the container exits."""
        resp = self.core.read_namespaced_pod_log(self.pod_name, self.ns, follow=True, _preload_content=False)
        buf = b""
        for chunk in resp.stream(1024):
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                yield line.decode(errors="replace")
        if buf:
            yield buf.decode(errors="replace")

    def exit_code(self) -> int | None:
        try:
            pod = self.core.read_namespaced_pod(self.pod_name, self.ns)
        except ApiException:
            return None
        for cs in pod.status.container_statuses or []:
            if cs.state.terminated is not None:
                return cs.state.terminated.exit_code
        return None

    def follow(self, out=sys.stdout, report=None) -> int:
        """Wait for the pod, stream its output, wait for the job to end; return the script's exit code."""
        report = report or (lambda m: print(m, file=sys.stderr, flush=True))
        last_code = None
        phase = self.wait_for_pod(report=report)
        if phase not in TERMINAL:
            report(f"kjob: {self.spec.name} running on {self.core.read_namespaced_pod(self.pod_name, self.ns).spec.node_name}")
            for line in self.stream_logs():
                m = re.match(r"\[kjob [^\]]*\] exit code (\d+)$", line)
                if m:
                    last_code = int(m.group(1))
                else:
                    print(line, file=out, flush=True)
        st = self.state()
        while st not in TERMINAL:  # pod finished; the operator updates the job status shortly after
            time.sleep(2)
            st = self.state()
        code = self.exit_code()
        code = code if code is not None else last_code
        report(f"kjob: {self.spec.name} {st} (exit code {code})")
        return code if code is not None else (0 if st == "Succeeded" else 1)

    def run(self, out=sys.stdout, cleanup: bool = True, report=None) -> int:
        """srun semantics: submit, stream output, wait, delete the job, return the exit code."""
        report = report or (lambda m: print(m, file=sys.stderr, flush=True))
        self.submit()
        report(f"kjob: submitted {self.spec.name} (log: {self.spec.log_path()})")
        try:
            return self.follow(out=out, report=report)
        except KeyboardInterrupt:
            report(f"kjob: interrupted; cancelling {self.spec.name}")
            return 130
        finally:
            if cleanup:
                self.delete()


# ---------- namespace-level helpers (used by the CLI) ----------
def job_state(obj: dict) -> str:
    conds = (obj.get("status") or {}).get("conditions") or []
    return conds[-1]["type"] if conds else "Pending"


def list_jobs(ns: str | None = None, everyone: bool = False) -> list[dict]:
    _load_config()
    ns = ns or current_namespace()
    sel = None if everyone else f"app.kubernetes.io/managed-by={MANAGED_BY}"
    return client.CustomObjectsApi().list_namespaced_custom_object(GROUP, VERSION, ns, PLURAL, label_selector=sel)["items"]


def delete_job(ns: str, name: str) -> bool:
    try:
        client.CustomObjectsApi().delete_namespaced_custom_object(
            GROUP, VERSION, ns, PLURAL, name, body=client.V1DeleteOptions(propagation_policy="Background"))
        return True
    except ApiException as e:
        if e.status == 404:
            return False
        raise


def job_events(ns: str, name: str, limit: int = 5) -> list[str]:
    try:
        evs = client.CoreV1Api().list_namespaced_event(ns, field_selector=f"involvedObject.name={name}").items
    except ApiException:
        return []
    evs.sort(key=lambda e: e.last_timestamp or e.event_time or _dt.datetime.min.replace(tzinfo=_dt.timezone.utc))
    return [f"{e.reason}: {e.message}" for e in evs[-limit:] if e.type == "Warning"]


def quota(ns: str | None = None) -> list[tuple[str, str, str]]:
    _load_config()
    ns = ns or current_namespace()
    rows = []
    for q in client.CoreV1Api().list_namespaced_resource_quota(ns).items:
        for k, hard in (q.status.hard or {}).items():
            rows.append((k, (q.status.used or {}).get(k, "0"), hard))
    return rows
