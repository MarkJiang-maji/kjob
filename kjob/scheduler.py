"""Fair release of queued jobs: decides which namespace's job starts next; Kubernetes still picks the node.

``ksub --fair`` creates the PyTorchJob with ``runPolicy.suspend: true``, so the operator makes no pod. Each pass hands
free capacity to the namespace holding the least of it, one job at a time, so queueing more jobs gains nothing. Usage
counts every pod that holds the resource, Notebooks included. Running jobs are never touched: re-suspending would
delete their pods. A released job keeps its capacity reserved until its pod is bound, or until RESERVE_SECONDS pass.
"""

from __future__ import annotations

import datetime as _dt
import time
from dataclasses import dataclass, field, replace

from kubernetes import client
from kubernetes.client.rest import ApiException
from kubernetes.utils.quantity import parse_quantity

from . import cluster, core

GPU = cluster.GPU
RELEASED = "kjob/released"      # annotation: when the job was released
RESERVE_SECONDS = 300
UNLIMITED = float("inf")


@dataclass
class Job:
    namespace: str
    name: str
    created: str
    requests: dict[str, float]
    pod_spec: dict = field(default_factory=dict, repr=False)
    raw: dict = field(default_factory=dict, repr=False)


@dataclass
class Node:
    name: str
    free: dict[str, float]
    labels: dict[str, str] = field(default_factory=dict)
    taints: list[dict] = field(default_factory=list)


def _tracked(resource: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(("cpu", "memory", GPU, resource)))


# ---------- does a job fit a node ----------
def _expression(e: dict, value: str | None) -> bool:
    op, values = e.get("operator"), e.get("values") or []
    if op == "In":
        return value in values
    if op == "NotIn":
        return value not in values
    if op == "Exists":
        return value is not None
    if op == "DoesNotExist":
        return value is None
    try:
        return value is not None and (int(value) > int(values[0]) if op == "Gt" else int(value) < int(values[0]))
    except (ValueError, IndexError):
        return False


def _selected(pod_spec: dict, node: Node) -> bool:
    if any(node.labels.get(k) != v for k, v in (pod_spec.get("nodeSelector") or {}).items()):
        return False
    required = (((pod_spec.get("affinity") or {}).get("nodeAffinity") or {})
                .get("requiredDuringSchedulingIgnoredDuringExecution") or {})
    terms = required.get("nodeSelectorTerms")
    return not terms or any(
        all(_expression(e, node.labels.get(e["key"])) for e in t.get("matchExpressions") or [])
        and all(_expression(e, node.name) for e in t.get("matchFields") or []) for t in terms)


def _tolerated(pod_spec: dict, taint: dict) -> bool:
    for t in pod_spec.get("tolerations") or []:
        if t.get("effect") not in (None, "", taint["effect"]) or t.get("key") not in (None, "", taint["key"]):
            continue
        if t.get("operator") == "Exists" or t.get("value", "") == taint["value"]:
            return True
    return False


def _best_node(job: Job, nodes: list[Node], resource: str) -> Node | None:
    """The tightest node the job fits, which leaves larger gaps for multi-GPU jobs."""
    fits = [n for n in nodes if all(n.free.get(k, 0) >= v for k, v in job.requests.items() if v > 0)
            and _selected(job.pod_spec, n) and all(_tolerated(job.pod_spec, t) for t in n.taints)]
    return min(fits, key=lambda n: (n.free.get(resource, 0), n.name), default=None)


def _take(job: Job, node: Node) -> None:
    for k, v in job.requests.items():
        node.free[k] = node.free.get(k, 0) - v


# ---------- the policy ----------
def pick_releases(queued: list[Job], nodes: list[Node], usage: dict[str, float] | None = None,
                  quota_left: dict[str, dict[str, float]] | None = None, budget: float = UNLIMITED,
                  resource: str = GPU) -> list[tuple[Job, str]]:
    """Jobs to release now, in order, with the node each was counted against (Kubernetes makes the real choice).

    Within a namespace the oldest job that fits goes first; a job that does not fit yet is skipped, so a job asking
    for many GPUs can keep waiting while smaller ones start.
    """
    nodes = [replace(n, free=dict(n.free)) for n in nodes]
    usage = dict(usage or {})
    quota_left = {ns: dict(q) for ns, q in (quota_left or {}).items()}
    waiting: dict[str, list[Job]] = {}
    for j in sorted(queued, key=lambda j: (j.created, j.name)):
        waiting.setdefault(j.namespace, []).append(j)

    def placement(job: Job):
        quota = quota_left.get(job.namespace, {})
        if job.requests.get(resource, 0) > budget or any(quota.get(k, UNLIMITED) < v for k, v in job.requests.items()):
            return None
        return _best_node(job, nodes, resource)

    picks = []
    while waiting:
        for ns in sorted(waiting, key=lambda n: (usage.get(n, 0), waiting[n][0].created, n)):
            hit = next(((j, n) for j, n in ((j, placement(j)) for j in waiting[ns]) if n), None)
            if hit:
                break
        else:
            return picks
        job, node = hit
        _take(job, node)
        amount = job.requests.get(resource, 0)
        usage[job.namespace] = usage.get(job.namespace, 0) + amount
        budget -= amount
        for k, v in job.requests.items():
            if k in quota_left.get(job.namespace, {}):
                quota_left[job.namespace][k] -= v
        waiting[job.namespace].remove(job)
        if not waiting[job.namespace]:
            del waiting[job.namespace]
        picks.append((job, node.name))
    return picks


# ---------- reading the cluster ----------
def _job(obj: dict, resource: str) -> Job | None:
    """The scheduler's view of a job; None for shapes it cannot place (kjob jobs are always a single Master pod)."""
    replicas = obj["spec"].get("pytorchReplicaSpecs") or {}
    if list(replicas) != ["Master"] or replicas["Master"].get("replicas", 1) != 1:
        return None
    pod_spec = replicas["Master"]["template"]["spec"]
    tracked = _tracked(resource)
    requests = dict.fromkeys(tracked, 0.0)
    for c in pod_spec.get("containers") or []:
        res = c.get("resources") or {}
        for k, v in {**(res.get("limits") or {}), **(res.get("requests") or {})}.items():
            if k in tracked:
                requests[k] += float(parse_quantity(v))
    meta = obj["metadata"]
    return Job(meta["namespace"], meta["name"], meta.get("creationTimestamp") or "", requests, pod_spec, obj)


def _is_queued(obj: dict) -> bool:
    return bool(obj["spec"].get("runPolicy", {}).get("suspend")) and not (obj.get("status") or {}).get("startTime") \
        and not obj["metadata"].get("deletionTimestamp")


def _is_active(obj: dict) -> bool:
    return not obj["spec"].get("runPolicy", {}).get("suspend") and core.job_state(obj) not in core.TERMINAL


def _still_reserved(obj: dict, now: _dt.datetime, reserve_seconds: int) -> bool:
    stamp = (obj["metadata"].get("annotations") or {}).get(RELEASED, "")
    try:
        released = _dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return False
    return (now - released).total_seconds() < reserve_seconds


def _nodes(items, pods, resource: str) -> list[Node]:
    tracked = _tracked(resource)
    used: dict[tuple[str, str], float] = {}
    for p in pods:
        if p.spec.node_name:
            for r in tracked:
                used[p.spec.node_name, r] = used.get((p.spec.node_name, r), 0.0) + cluster._requests(p, r)
    out = []
    for n in items:
        alloc = n.status.allocatable or {}
        ready = any(c.type == "Ready" and c.status == "True" for c in n.status.conditions or [])
        if not ready or n.spec.unschedulable or float(parse_quantity(alloc.get(resource, "0"))) <= 0:
            continue
        name = n.metadata.name
        out.append(Node(name, {r: float(parse_quantity(alloc.get(r, "0"))) - used.get((name, r), 0.0) for r in tracked},
                        dict(n.metadata.labels or {}),
                        [{"key": t.key, "value": t.value or "", "effect": t.effect} for t in n.spec.taints or []
                         if t.effect in ("NoSchedule", "NoExecute")]))
    return out


def _quota_left(quotas) -> dict[str, dict[str, float]]:
    left: dict[str, dict[str, float]] = {}
    for q in quotas:
        room = left.setdefault(q.metadata.namespace, {})
        used = q.status.used or {}
        for key, hard in (q.status.hard or {}).items():
            name = key[len("requests."):] if key.startswith("requests.") else key
            rest = float(parse_quantity(hard)) - float(parse_quantity(used.get(key, "0")))
            room[name] = min(room.get(name, UNLIMITED), rest)
    return left


def _release_patch(obj: dict, now: _dt.datetime) -> dict:
    """Merge patch, which is what every supported client sends; resourceVersion makes it fail (409) if the job changed."""
    return {"metadata": {"resourceVersion": obj["metadata"]["resourceVersion"],
                         "annotations": {RELEASED: now.strftime("%Y-%m-%dT%H:%M:%SZ")}},
            "spec": {"runPolicy": {"suspend": False}}}


# ---------- one pass, and the loop ----------
def reconcile(custom=None, core_api=None, resource: str = GPU, namespaces: list[str] | None = None,
              max_in_use: float | None = None, reserve_seconds: int = RESERVE_SECONDS, dry_run: bool = False,
              report=print) -> list[tuple[Job, str]]:
    """One scheduling pass. namespaces limits which jobs it manages; usage and free capacity always cover all pods."""
    custom = custom or client.CustomObjectsApi()
    core_api = core_api or client.CoreV1Api()
    objs = custom.list_cluster_custom_object(core.GROUP, core.VERSION, core.PLURAL,
                                             label_selector=f"{core.FAIR_LABEL}={core.FAIR_QUEUE}")["items"]
    if namespaces:
        objs = [o for o in objs if o["metadata"]["namespace"] in namespaces]
    queued = []
    for o in objs:
        job = _job(o, resource) if _is_queued(o) else None
        if job:
            queued.append(job)
        elif _is_queued(o):
            report(f"kjob: skipping {o['metadata']['namespace']}/{o['metadata']['name']}: not a single Master pod")
    if not queued:
        return []

    pods = core_api.list_pod_for_all_namespaces(field_selector="status.phase!=Succeeded,status.phase!=Failed").items
    nodes = _nodes(core_api.list_node().items, pods, resource)
    quota_left = _quota_left(core_api.list_resource_quota_for_all_namespaces().items)
    existing = {(p.metadata.namespace, p.metadata.name): p for p in pods}
    usage: dict[str, float] = {}
    for p in pods:
        if p.spec.node_name:
            usage[p.metadata.namespace] = usage.get(p.metadata.namespace, 0.0) + cluster._requests(p, resource)

    now = _dt.datetime.now(_dt.timezone.utc)
    in_use = 0.0
    for o in objs:
        job = _job(o, resource) if _is_active(o) else None
        if not job:
            continue
        pod = existing.get((job.namespace, f"{job.name}-master-0"))
        bound = bool(pod and pod.spec.node_name)
        if not bound and not _still_reserved(o, now, reserve_seconds):
            continue
        in_use += job.requests.get(resource, 0)
        if bound:
            continue     # already counted through its pod
        usage[job.namespace] = usage.get(job.namespace, 0.0) + job.requests.get(resource, 0)
        node = _best_node(job, nodes, resource)
        if node:
            _take(job, node)
        if not pod:      # a created pod already counts in the quota's used
            room = quota_left.get(job.namespace, {})
            for k, v in job.requests.items():
                if k in room:
                    room[k] -= v

    budget = UNLIMITED if max_in_use is None else max_in_use - in_use
    picks = pick_releases(queued, nodes, usage, quota_left, budget, resource)
    released = []
    for job, node in picks:
        what = f"{job.namespace}/{job.name} ({job.requests.get(resource, 0):g} {resource}, counted on {node})"
        if dry_run:
            report(f"kjob: would release {what}")
            released.append((job, node))
            continue
        try:
            custom.patch_namespaced_custom_object(core.GROUP, core.VERSION, job.namespace, core.PLURAL, job.name,
                                                  _release_patch(job.raw, now))
        except ApiException as e:     # deleted or changed meanwhile: the next pass sees the new state
            report(f"kjob: could not release {job.namespace}/{job.name}: {e.status} {e.reason} {(e.body or '')[:200]}")
            continue
        report(f"kjob: released {what}")
        released.append((job, node))
    return released


def run(interval: int = 15, report=print, **options) -> None:
    while True:
        try:
            reconcile(report=report, **options)
        except Exception as e:  # keep scheduling after transient API errors
            report(f"kjob: fair scheduler pass failed: {type(e).__name__}: {e}")
        time.sleep(interval)
