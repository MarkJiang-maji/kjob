"""Submit, monitor and clean up Kubeflow PyTorchJobs that mirror the current notebook's environment.

The job copies the notebook pod's image, working directory, security context and the volumes mounted into the
notebook container, so code, virtualenvs and data on those volumes are visible at the same paths. Volumes that a
namespace PodDefault injects are left out (the PodDefault webhook adds them again and rejects duplicates).
ReadWriteOnce claims that are backed by NFS are mounted as the underlying NFS export so jobs can run on other nodes.
"""

from __future__ import annotations

import datetime as _dt
import copy
import json
import os
import re
import shlex
import shutil
import socket
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
from kubernetes.utils.quantity import parse_quantity

from . import cluster

GROUP, VERSION, PLURAL = "kubeflow.org", "v1", "pytorchjobs"
MANAGED_BY = "kjob"
SIDECAR_CONTAINERS = {"istio-proxy"}
GPU_PRODUCTS = {  # short names for --gpu-type; any other value is used as the raw nvidia.com/gpu.product label
    "a5000": "NVIDIA-RTX-A5000",
    "v100": "Tesla-V100-SXM2-16GB",
    "a30": "NVIDIA-A30",                 # whole GPU
    "a30-shared": "NVIDIA-A30-SHARED",   # MPS slice; still counts as one nvidia.com/gpu in the quota
}
# NFS volumes written by the GroupShare / NamespaceShare PodDefaults. A notebook started before its PodDefault changed
# keeps the old ones (e.g. a group the user has since left); the PodDefault adds the current ones to every job.
MANAGED_VOLUME_PREFIXES = ("gs-", "ns-")
# On MPS nodes (product label "...-SHARED", sharing-strategy=mps) the device plugin mounts its own /dev/shm into GPU
# containers; a second /dev/shm volume makes the container fail with "Duplicate mount point: /dev/shm".
SHARING_STRATEGY_LABEL = "nvidia.com/gpu.sharing-strategy"
TERMINAL = ("Succeeded", "Failed", "Deleted")  # Deleted: the job is gone (kcancel elsewhere, TTL)
DEFAULT_CPUS, DEFAULT_MEMORY, DEFAULT_SHM = 1, "2Gi", "256Mi"


def unique_name(stem: str) -> str:
    try:
        base = sanitize_name(stem)[:37].rstrip("-")
    except ValueError:      # e.g. a script name without ASCII letters or digits
        base = "job"
    return f"{base}-{uuid.uuid4().hex[:12]}"


def _matches(selector: dict, labels: dict) -> bool:
    if any(labels.get(k) != v for k, v in selector.get("matchLabels", {}).items()):
        return False
    for e in selector.get("matchExpressions", []):
        key, op, values = e["key"], e["operator"], e.get("values", [])
        if op == "In" and labels.get(key) not in values:
            return False
        if op == "NotIn" and key in labels and labels[key] in values:
            return False
        if op == "Exists" and key not in labels:
            return False
        if op == "DoesNotExist" and key in labels:
            return False
        if op not in {"In", "NotIn", "Exists", "DoesNotExist"}:
            raise ValueError(f"Unsupported PodDefault selector operator: {op}")
    return True


def _load_config() -> None:
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()


def current_namespace() -> str:
    f = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
    return f.read_text().strip() if f.exists() else os.environ.get("NB_NAMESPACE", "default")


def current_notebook() -> str:
    """Name of the notebook pod this code runs in (the job label kjob/notebook)."""
    return socket.gethostname()


def sanitize_name(name: str) -> str:
    """Job names: lowercase alphanumerics and '-', starting with a letter (the operator requires a DNS-1035 label),
    at most 50 chars (pods add '-master-0')."""
    n = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    if not n:
        raise ValueError(f"invalid job name {name!r}")
    if not n[0].isalpha():
        n = "j-" + n
    return n[:50].rstrip("-")


def _nfs_mounts() -> dict[str, tuple[str, str]]:
    """mount point -> (server, export path) for NFS mounts visible in this container."""
    out = {}
    for line in Path("/proc/mounts").read_text().splitlines():
        src, mnt, fstype = [re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), x)
                            for x in line.split()[:3]]
        if fstype.startswith("nfs") and ":" in src:
            server, path = src.rsplit(":", 1)
            out[mnt] = (server, path)
    return out


def job_shell(name: str, command: str, workdir: str, log: str) -> str:
    """Container command. Output goes to the pod log and to the log file. The script runs in its own process group
    and gets the SIGTERM sent at the -t limit or by kcancel, so it can save state within the grace period
    (bash as PID 1 would otherwise ignore SIGTERM and everything is killed 30 s later)."""
    q, ts = shlex.quote, "$(date -u +%FT%TZ)"
    return (
        "set -o pipefail\n"
        f"mkdir -p {q(str(Path(log).parent))}\n"
        f"cd {q(workdir)} || exit 1\n"
        'state="${TMPDIR:-/tmp}/kjob-$$"\n'
        f"log={q(log)}\n"
        f"trap 'echo \"[kjob {ts}] stopping: SIGTERM (time limit or kcancel)\" | tee -a \"$log\"; "
        'p=$(cat "$state.pid" 2>/dev/null); [ -n "$p" ] && { kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p"; }\' TERM INT\n'
        f'{{ echo "[kjob {ts}] start {name} on node $KJOB_NODE"\n'
        f"  if command -v setsid >/dev/null 2>&1; then setsid {command} & else {command} & fi\n"
        '  echo $! > "$state.pid"; wait $!; rc=$?\n'
        f'  echo "[kjob {ts}] exit code $rc"; echo $rc > "$state.rc"; }} 2>&1 | tee -a "$log" &\n'
        "tee_pid=$!\n"
        "while kill -0 $tee_pid 2>/dev/null; do wait $tee_pid; done\n"
        'exit "$(cat "$state.rc" 2>/dev/null || echo 1)"\n')


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
    container_security_context: object = None
    effective_uid: int = field(default_factory=os.getuid)
    job_labels: dict = field(default_factory=dict)
    image_pull_secrets: list = field(default_factory=list)
    skipped: dict = field(default_factory=dict)          # volume name -> reason

    @classmethod
    def from_current_pod(cls, shm_size: str = DEFAULT_SHM) -> "NotebookTemplate":
        _load_config()
        ns = current_namespace()
        core = client.CoreV1Api()
        pod = core.read_namespaced_pod(current_notebook(), ns)
        main = [c for c in pod.spec.containers if c.name not in SIDECAR_CONTAINERS][0]
        injected = set()
        job_labels = {}
        labels = pod.metadata.labels or {}
        try:
            for pd in client.CustomObjectsApi().list_namespaced_custom_object("kubeflow.org", "v1alpha1", ns, "poddefaults")["items"]:
                selector = pd["spec"].get("selector", {})
                if not _matches(selector, labels):
                    continue
                injected |= {v["name"] for v in pd["spec"].get("volumes", [])}
                keys = set(selector.get("matchLabels", {})) | {e["key"] for e in selector.get("matchExpressions", [])}
                job_labels.update({k: labels[k] for k in keys if k in labels})
        except ApiException as e:
            if e.status not in (403, 404):
                raise
        mounts = {m.name: m for m in main.volume_mounts or []}
        nfs = _nfs_mounts()
        identity = {k: getattr(main.security_context, k, None)
                    for k in ("run_as_user", "run_as_group", "run_as_non_root")}
        identity = {k: v for k, v in identity.items() if v is not None}
        t = cls(namespace=ns, pod_name=pod.metadata.name, node=pod.spec.node_name, image=main.image,
                working_dir=os.getcwd(), security_context=pod.spec.security_context,
                container_security_context=client.V1SecurityContext(**identity) if identity else None,
                effective_uid=identity.get("run_as_user", getattr(pod.spec.security_context, "run_as_user", None)
                                           if pod.spec.security_context else None),
                job_labels=job_labels,
                image_pull_secrets=pod.spec.image_pull_secrets or [])
        if t.effective_uid is None:
            t.effective_uid = os.getuid()
        for v in pod.spec.volumes or []:
            m = mounts.get(v.name)
            if m is None:
                t.skipped[v.name] = "not mounted in the notebook container"
            elif v.name in injected:
                t.skipped[v.name] = "added automatically by a PodDefault"
            elif v.name.startswith(MANAGED_VOLUME_PREFIXES):
                t.skipped[v.name] = "outdated GroupShare/NamespaceShare mount (restart the notebook to refresh it)"
            elif v.projected is not None or v.name.startswith("kube-api-access"):
                t.skipped[v.name] = "service-account token"
            elif v.persistent_volume_claim is not None and m.mount_path in nfs:
                if m.sub_path or m.sub_path_expr:
                    raise ValueError(f"NFS-backed PVC {v.name} uses subPath/subPathExpr; mount the full PVC before submitting.")
                server, path = nfs[m.mount_path]
                readonly = bool(v.persistent_volume_claim.read_only or m.read_only)
                t.volumes.append(client.V1Volume(name=v.name, nfs=client.V1NFSVolumeSource(server=server, path=path, read_only=readonly)))
                t.volume_mounts.append(client.V1VolumeMount(name=v.name, mount_path=m.mount_path, read_only=readonly))
            elif v.empty_dir is not None:
                medium = v.empty_dir.medium
                t.volumes.append(client.V1Volume(name=v.name, empty_dir=client.V1EmptyDirVolumeSource(
                    medium=medium, size_limit=shm_size if medium == "Memory" else None)))
                t.volume_mounts.append(copy.deepcopy(m))
            else:  # PVC on non-NFS storage, nfs, secret, configMap, ...: reuse as-is
                t.volumes.append(v)
                t.volume_mounts.append(copy.deepcopy(m))
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
    cpus: int = DEFAULT_CPUS
    memory: str = DEFAULT_MEMORY
    shm: str = DEFAULT_SHM
    gpu_types: list[str] = field(default_factory=list)   # e.g. ["a5000", "v100"]
    node: str | None = None                              # pin to one host
    exclude_nodes: list[str] = field(default_factory=list)
    interpreter: str | None = None                       # default: this Python for .py, bash for .sh
    env: dict[str, str] = field(default_factory=dict)
    workdir: str | None = None
    log_dir: str = str(Path.home() / "kjob-logs")
    ttl_seconds: int = 3600                              # safety net if nobody monitors the job
    time_limit: int | None = None                        # seconds; the job is stopped after this (like --time)

    def validate(self) -> None:
        if not isinstance(self.gpus, int) or self.gpus < 0:
            raise ValueError("GPU count must be a nonnegative integer.")
        if not isinstance(self.cpus, int) or self.cpus < 1:
            raise ValueError("CPU count must be a positive integer.")
        if not isinstance(self.ttl_seconds, int) or self.ttl_seconds < 0:
            raise ValueError("TTL must be a nonnegative integer.")
        if self.time_limit is not None and (not isinstance(self.time_limit, int) or self.time_limit < 1):
            raise ValueError("Time limit must be a positive number of seconds.")
        for key, value in (("memory", self.memory), ("shared memory", self.shm)):
            try:
                amount = parse_quantity(value)
                if not amount.is_finite() or amount <= 0:
                    raise ValueError()
            except (ValueError, TypeError, ArithmeticError):
                raise ValueError(f"Invalid {key} quantity: {value!r}") from None
        products = [GPU_PRODUCTS.get(g, g) for g in self.gpu_types]
        shared = [p.endswith("-SHARED") for p in products]
        if any(shared) and not all(shared):
            raise ValueError("Choose either whole GPUs or shared GPUs; do not mix them in --gpu-type.")
        if self.gpus > 1 and any(shared):
            raise ValueError("MPS supports only one shared GPU unit per container; use a whole GPU type for -G > 1.")
        if not self.gpus and self.gpu_types:
            raise ValueError("CPU-only jobs cannot select a GPU type.")
        protected = {"KJOB_NAME", "KJOB_NODE", "PYTHONUNBUFFERED", "NVIDIA_VISIBLE_DEVICES", "CUDA_MPS_PIPE_DIRECTORY",
                     "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"}
        if not self.gpus:
            protected.add("CUDA_VISIBLE_DEVICES")
        if any(shared):
            protected.add("CUDA_VISIBLE_DEVICES")
        if protected.intersection(self.env):
            raise ValueError("Reserved job environment variables cannot be overridden: " + ", ".join(sorted(protected.intersection(self.env))))
        for k in self.env:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
                raise ValueError(f"Invalid environment variable name: {k!r}")

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
        spec.validate()
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
        s.validate()
        workdir = s.workdir or t.working_dir
        log = s.log_path()
        shell = job_shell(s.name, s.command(), workdir, log)
        res = {"cpu": str(s.cpus), "memory": s.memory}
        if s.gpus:
            res["nvidia.com/gpu"] = str(s.gpus)  # CPU-only jobs must not request GPUs at all
        env = [client.V1EnvVar(name="PYTHONUNBUFFERED", value="1"),
               client.V1EnvVar(name="KJOB_NAME", value=s.name),
               client.V1EnvVar(name="KJOB_NODE", value_from=client.V1EnvVarSource(
                   field_ref=client.V1ObjectFieldSelector(field_path="spec.nodeName")))]
        if not s.gpus:  # CUDA images may default NVIDIA_VISIBLE_DEVICES=all, exposing the node's GPUs without a request
            hide = {"NVIDIA_VISIBLE_DEVICES": "void", "CUDA_VISIBLE_DEVICES": "-1"}
            env += [client.V1EnvVar(name=k, value=v) for k, v in hide.items()]
        env += [client.V1EnvVar(name=k, value=v) for k, v in s.env.items()]
        products = [GPU_PRODUCTS.get(g, g) for g in s.gpu_types]
        mps = bool(s.gpus) and any(p.endswith("-SHARED") for p in products)
        if mps and t.effective_uid != 0:
            raise ValueError("This installation only enables shared MPS jobs for UID 0 Notebooks; the administrator must also verify the server UID and health. Use a whole GPU type for a non-root Notebook.")
        volumes, volume_mounts = t.volumes, t.volume_mounts
        if mps:  # the device plugin provides /dev/shm
            shm = {m.name for m in t.volume_mounts if m.mount_path == "/dev/shm"}
            volumes = [v for v in t.volumes if v.name not in shm]
            volume_mounts = [m for m in t.volume_mounts if m.name not in shm]
        container = client.V1Container(
            name="pytorch", image=t.image, command=["bash", "-lc", shell], working_dir=workdir, env=env,
            security_context=t.container_security_context,
            resources=client.V1ResourceRequirements(requests=res, limits=res), volume_mounts=volume_mounts)
        terms = []
        if s.node:
            terms.append(client.V1NodeSelectorRequirement(key="kubernetes.io/hostname", operator="In", values=[s.node]))
        if s.gpu_types:
            terms.append(client.V1NodeSelectorRequirement(key="nvidia.com/gpu.product", operator="In", values=products))
        if s.gpus and not mps:
            terms.append(client.V1NodeSelectorRequirement(key=SHARING_STRATEGY_LABEL, operator="NotIn", values=["mps"]))
        elif mps:
            terms.append(client.V1NodeSelectorRequirement(key=SHARING_STRATEGY_LABEL, operator="In", values=["mps"]))
        if s.exclude_nodes:
            terms.append(client.V1NodeSelectorRequirement(key="kubernetes.io/hostname", operator="NotIn", values=s.exclude_nodes))
        affinity = client.V1Affinity(node_affinity=client.V1NodeAffinity(
            required_during_scheduling_ignored_during_execution=client.V1NodeSelector(
                node_selector_terms=[client.V1NodeSelectorTerm(match_expressions=terms)]))) if terms else None
        pod_spec = client.V1PodSpec(containers=[container], volumes=volumes, affinity=affinity,
                                    security_context=t.security_context, image_pull_secrets=t.image_pull_secrets or None,
                                    automount_service_account_token=False,  # the script does not need API access
                                    # Enforced by the kubelet from Pod start, so queueing time does not count
                                    # (runPolicy.activeDeadlineSeconds is only checked when the operator reconciles).
                                    active_deadline_seconds=s.time_limit)
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(annotations={"sidecar.istio.io/inject": "false"},
                                         labels={**t.job_labels, "app.kubernetes.io/managed-by": MANAGED_BY}),
            spec=pod_spec)
        api = client.ApiClient()
        return {
            "apiVersion": f"{GROUP}/{VERSION}", "kind": "PyTorchJob",
            "metadata": {"name": s.name, "namespace": self.ns,
                         "labels": {"app.kubernetes.io/managed-by": MANAGED_BY, "kjob/notebook": t.pod_name},
                         "annotations": {"kjob/command": s.command()[:2000], "kjob/log": log,
                                         "kjob/workdir": workdir,
                                         "kjob/submitted": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}},
            "spec": {
                # RestartPolicy Never already makes script failures terminal. A zero backoffLimit
                # on older operators deletes a failed Pod before recording Failed and can recreate it.
                "runPolicy": {"cleanPodPolicy": "None", "ttlSecondsAfterFinished": s.ttl_seconds},
                "pytorchReplicaSpecs": {"Master": {"replicas": 1, "restartPolicy": "Never",
                                                   "template": api.sanitize_for_serialization(template)}},
            },
        }

    # ---------- lifecycle ----------
    def submit(self) -> str:
        if self.spec.gpus and self.spec.node and not self.spec.gpu_types:
            labels = self._node_labels(self.spec.node)
            if labels.get(SHARING_STRATEGY_LABEL) == "mps":
                product = labels.get("nvidia.com/gpu.product", "")
                if not product.endswith("-SHARED"):
                    raise ValueError("MPS node does not advertise a -SHARED product label; ask the administrator to configure GPU selection.")
                self.spec.gpu_types = [product]
        self.spec.validate()
        try:
            quotas = self.core.list_namespaced_resource_quota(self.ns).items
        except ApiException as e:
            if e.status != 403:
                raise
            quotas = []
        requested = {"requests.cpu": self.spec.cpus, "requests.memory": self.spec.memory,
                     "requests.nvidia.com/gpu": self.spec.gpus}
        for q in quotas:
            hard = q.status.hard or {}
            for key, value in requested.items():
                if key in hard and parse_quantity(value) > parse_quantity(hard[key]):
                    raise ValueError(f"Request {key}={value} exceeds namespace quota {q.metadata.name} hard limit {hard[key]}; waiting cannot satisfy it.")
        self._warn_unshared_paths()
        obj = self.custom.create_namespaced_custom_object(GROUP, VERSION, self.ns, PLURAL, self.manifest())
        self.uid = obj["metadata"]["uid"]
        record_history(self)
        return self.spec.name

    def _node_labels(self, name: str) -> dict:
        """Labels of a pinned node: read directly, else from the published cluster status, else assume a whole-GPU node."""
        try:
            return self.core.read_node(name).metadata.labels or {}
        except ApiException as e:
            if e.status != 403:
                raise
        try:
            node = next((n for n in cluster.load_snapshot(self.core)["nodes"] if n["name"] == name), None)
        except (cluster.ClusterViewUnavailable, ApiException, ValueError):
            node = None
        if node is None:
            print(f"kjob: cannot inspect node {name}; requesting whole GPUs there (use -g ...-shared for an MPS node)",
                  file=sys.stderr)
            return {}
        labels = {SHARING_STRATEGY_LABEL: node["sharing"]} if node["sharing"] else {}
        if node["product"]:
            labels["nvidia.com/gpu.product"] = node["product"]
        return labels

    def _warn_unshared_paths(self) -> None:
        shared = self.t.shared_paths()
        script = Path(self.spec.script).expanduser().resolve()
        paths = [self.spec.workdir or self.t.working_dir, self.spec.log_dir]
        if script.is_file():
            paths.append(str(script))
        elif "/" in self.spec.script or not shutil.which(self.spec.script):
            raise ValueError(f"Script is not a regular file: {script}")
        # else: a command on PATH (e.g. nvidia-smi); the job uses the same image, so it is there too
        for p in paths:
            resolved = str(Path(p).expanduser().resolve())
            covering = [m for m in self.t.volume_mounts if resolved == m.mount_path or
                        resolved.startswith(m.mount_path.rstrip("/") + "/")]
            closest = max(covering, key=lambda m: len(m.mount_path), default=None)
            if not closest or closest.mount_path not in shared:
                raise ValueError(f"{resolved} is not on a shared volume; place scripts, working directory and logs on the Notebook workspace.")
        log = str(Path(self.spec.log_dir).expanduser().resolve())
        covering = max((m for m in self.t.volume_mounts if log == m.mount_path or
                        log.startswith(m.mount_path.rstrip("/") + "/")), key=lambda m: len(m.mount_path))
        volume = next(v for v in self.t.volumes if v.name == covering.name)
        source = volume.nfs or volume.persistent_volume_claim
        if covering.read_only or (source is not None and source.read_only):
            raise ValueError("Log directory is on a read-only volume; choose a writable workspace.")

    def state(self) -> str:
        try:
            obj = self.custom.get_namespaced_custom_object(GROUP, VERSION, self.ns, PLURAL, self.spec.name)
            uid = obj["metadata"]["uid"]
            if getattr(self, "uid", uid) != uid:
                return "Deleted"
            self.uid = uid
            return job_state(obj)
        except ApiException as e:
            if e.status == 404:
                return "Deleted"
            raise

    def delete(self) -> None:
        if not getattr(self, "uid", None) and self.state() == "Deleted":
            return
        delete_job(self.ns, self.spec.name, uid=self.uid)

    def _pod(self):
        pod = self.core.read_namespaced_pod(self.pod_name, self.ns)
        if getattr(self, "uid", None) and not any(o.uid == self.uid and o.kind == "PyTorchJob"
                                                 for o in pod.metadata.owner_references or []):
            raise ApiException(status=404, reason="Pod belongs to a different job UID")
        return pod

    def wait_for_pod(self, timeout: float | None = None, report=print) -> str:
        """Block until the pod leaves Pending (or the job ends); report scheduling problems while waiting."""
        t0, seen = time.time(), set()
        while True:
            st = self.state()
            if st == "Deleted":
                return st
            try:
                pod = self._pod()
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
                for ev in job_events(self.ns, self.spec.name, uid=getattr(self, "uid", None)):
                    if ev not in seen:
                        seen.add(ev)
                        report(f"kjob: {self.spec.name} event: {ev}")
                    if "FailedValidation" in ev.split(":", 1)[0]:
                        return "Invalid"      # the operator rejected the job; it will never get a pod
            if st in TERMINAL:
                return st
            if timeout is not None and time.time() - t0 > timeout:
                return "Pending"
            time.sleep(3)

    def stream_logs(self) -> Iterator[str]:
        """Yield log lines in real time until the container exits."""
        self._pod()
        resp = self.core.read_namespaced_pod_log(self.pod_name, self.ns, follow=True, _preload_content=False)
        buf = b""
        try:
            for chunk in resp.stream(1024):
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    yield line.decode(errors="replace")
            if buf:
                yield buf.decode(errors="replace")
        finally:
            resp.close()

    def stop_reason(self) -> str | None:
        """Why the pod ended, when the pod says so (DeadlineExceeded at the -t limit)."""
        try:
            return self._pod().status.reason
        except ApiException:
            return None

    def exit_code(self) -> int | None:
        try:
            pod = self._pod()
        except ApiException as e:
            if e.status == 404:
                return None
            raise
        for cs in pod.status.container_statuses or []:
            if cs.state.terminated is not None:
                self.exit_reason = cs.state.terminated.reason  # e.g. OOMKilled, Error, Completed
                return cs.state.terminated.exit_code
        return None

    def follow(self, out=sys.stdout, report=None) -> int:
        """Wait for the pod, stream its output, wait for the job to end; return the script's exit code."""
        report = report or (lambda m: print(m, file=sys.stderr, flush=True))
        last_code = None
        phase = self.wait_for_pod(report=report)
        if phase == "Invalid":
            report(f"kjob: {self.spec.name} was rejected by the training operator (see the event above)")
            return 1
        if phase != "Deleted":
            try:
                report(f"kjob: {self.spec.name} running on {self._pod().spec.node_name}")
                for line in self.stream_logs():
                    m = re.match(r"\[kjob [^\]]*\] exit code (\d+)$", line)
                    if m:
                        last_code = int(m.group(1))
                    else:
                        print(line, file=out, flush=True)
            except ApiException as e:
                if e.status != 404:
                    raise
                report(f"kjob: {self.spec.name} pod disappeared; checking job status")
        st = self.state()
        while st not in TERMINAL:  # pod finished; the operator updates the job status shortly after
            time.sleep(2)
            st = self.state()
        code = self.exit_code()
        code = code if code is not None else last_code
        reason = self.stop_reason() or getattr(self, "exit_reason", None)
        detail = f", {reason}" if reason and reason not in ("Completed", "Error") else ""
        report(f"kjob: {self.spec.name} {st} (exit code {code}{detail})")
        return code if code is not None else (0 if st == "Succeeded" else 1)

    def run(self, out=sys.stdout, cleanup: bool = True, report=None) -> int:
        """srun semantics: submit, stream output, wait, delete the job, return the exit code."""
        report = report or (lambda m: print(m, file=sys.stderr, flush=True))
        self.submit()
        report(f"kjob: submitted {self.spec.name} (log: {self.spec.log_path()})")
        try:
            code = self.follow(out=out, report=report)
        except KeyboardInterrupt:
            report(f"kjob: interrupted; cancelling {self.spec.name}")
            if cleanup:
                self.delete()
            return 130
        if cleanup:
            self.delete()
        return code


HISTORY_FILE = Path.home() / ".kjob" / "history.jsonl"


def record_history(job: "KJob") -> None:
    """Append the submitted job to ~/.kjob/history.jsonl (read by `kjob acct`); failures are ignored."""
    s = job.spec
    rec = {"name": s.name, "namespace": job.ns, "submitted": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "gpus": s.gpus, "gpu_types": s.gpu_types, "node": s.node, "cpus": s.cpus, "memory": s.memory,
           "time_limit": s.time_limit, "command": s.command()[:500], "log": s.log_path()}
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with HISTORY_FILE.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def read_history() -> list[dict]:
    try:
        lines = HISTORY_FILE.read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def log_runs(path: str) -> list[dict]:
    """Runs recorded in a job log by the wrapper's [kjob ...] lines: start, end, exit code, stopped (SIGTERM)."""
    runs = []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                m = re.match(r"\[kjob (\S+)\] (start|exit code (\d+)|stopping)", line)
                if not m:
                    continue
                if m.group(2) == "start" or not runs:
                    runs.append({"start": m.group(1) if m.group(2) == "start" else None, "end": None, "code": None,
                                 "stopped": False})
                if m.group(2) == "stopping":
                    runs[-1]["stopped"], runs[-1]["end"] = True, m.group(1)
                elif m.group(3):
                    runs[-1]["end"], runs[-1]["code"] = m.group(1), int(m.group(3))
    except OSError:
        pass
    return runs


def log_result(path: str) -> tuple[str | None, str | None, int | None]:
    """(start, end, exit code) of the last run in a job log."""
    runs = log_runs(path)
    return (runs[-1]["start"], runs[-1]["end"], runs[-1]["code"]) if runs else (None, None, None)


# ---------- namespace-level helpers (used by the CLI) ----------
def job_state(obj: dict) -> str:
    conds = (obj.get("status") or {}).get("conditions") or []
    active = [c["type"] for c in conds if str(c.get("status", "False")).lower() == "true"]
    for terminal in ("Failed", "Succeeded"):
        if terminal in active:
            return terminal
    return active[-1] if active else "Pending"


def list_jobs(ns: str | None = None, everyone: bool = False, notebook: str | None = None) -> list[dict]:
    """kjob jobs in the namespace (everyone: all PyTorchJobs); notebook: only jobs submitted from that notebook pod."""
    _load_config()
    ns = ns or current_namespace()
    sel = None if everyone else f"app.kubernetes.io/managed-by={MANAGED_BY}"
    if notebook and not everyone:
        sel += f",kjob/notebook={notebook}"
    return client.CustomObjectsApi().list_namespaced_custom_object(GROUP, VERSION, ns, PLURAL, label_selector=sel)["items"]


def delete_job(ns: str, name: str, uid: str | None = None) -> bool:
    api = client.CustomObjectsApi()
    try:
        if not uid:
            uid = api.get_namespaced_custom_object(GROUP, VERSION, ns, PLURAL, name)["metadata"]["uid"]
        api.delete_namespaced_custom_object(
            GROUP, VERSION, ns, PLURAL, name,
            body=client.V1DeleteOptions(propagation_policy="Background", preconditions=client.V1Preconditions(uid=uid)))
        return True
    except ApiException as e:
        if e.status in (404, 409):
            return False
        raise


def job_events(ns: str, name: str, limit: int = 5, uid: str | None = None) -> list[str]:
    """Recent warnings about the job; uid excludes events left by an earlier job with the same name."""
    sel = f"involvedObject.name={name}" + (f",involvedObject.uid={uid}" if uid else "")
    try:
        evs = client.CoreV1Api().list_namespaced_event(ns, field_selector=sel).items
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
