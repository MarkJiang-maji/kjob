"""Tests for the cluster view, history, time limit and script directives (no cluster needed)."""
import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kubernetes import client

from kjob import cli, cluster, core
from test_kjob import manifest


def k8s(obj, kind):
    return client.ApiClient()._ApiClient__deserialize(obj, kind)


def node(name, gpus, product=None, taint=None):
    labels = {"nvidia.com/gpu.product": product} if product else {}
    return k8s({"metadata": {"name": name, "labels": labels},
                "spec": {"taints": [{"key": taint, "value": "x", "effect": "NoSchedule"}] if taint else None},
                "status": {"allocatable": {"nvidia.com/gpu": str(gpus), "cpu": "64", "memory": "256Gi"},
                           "conditions": [{"type": "Ready", "status": "True"}]}}, "V1Node")


def pod(ns, name, gpus, node_name=None, phase="Running", labels=None, owner=None, affinity_host=None):
    spec = {"containers": [{"name": "c", "image": "i", "resources": {"requests": {"nvidia.com/gpu": str(gpus), "cpu": "2"}}}],
            "nodeName": node_name}
    if affinity_host:
        spec["affinity"] = {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": [
            {"matchExpressions": [{"key": "kubernetes.io/hostname", "operator": "In", "values": [affinity_host]}]}]}}}
    meta = {"namespace": ns, "name": name, "labels": labels or {}, "creationTimestamp": "2026-01-01T00:00:00Z"}
    if owner:
        meta["ownerReferences"] = [{"apiVersion": "kubeflow.org/v1", "kind": "PyTorchJob", "name": owner, "uid": "u"}]
    status = {"phase": phase}
    if phase == "Pending":
        status["conditions"] = [{"type": "PodScheduled", "status": "False", "message": "0/8 nodes are available"}]
    return k8s({"metadata": meta, "spec": spec, "status": status}, "V1Pod")


def fake_api():
    api = mock.Mock()
    api.list_node.return_value.items = [node("a5000-1", 10, "NVIDIA-RTX-A5000"), node("rtxpro6000", 8, taint="restricted-gpu"),
                                        node("cpu-1", 0)]
    api.list_pod_for_all_namespaces.return_value.items = [
        pod("alice", "nb-0", 2, "a5000-1", labels={"notebook-name": "nb"}),
        pod("bob", "train-master-0", 4, "rtxpro6000", labels={"app.kubernetes.io/managed-by": "kjob"}, owner="train"),
        pod("carol", "big-0", 8, phase="Pending", labels={"notebook-name": "big"}, affinity_host="rtxpro6000"),
        pod("dave", "cpu-pod", 0, "cpu-1"),
    ]
    return api


class SnapshotTest(unittest.TestCase):
    def test_counts_gpus_per_node_and_classifies_workloads(self):
        snap = cluster.build_snapshot(fake_api())
        nodes = {n["name"]: n for n in snap["nodes"]}
        self.assertEqual(set(nodes), {"a5000-1", "rtxpro6000"})            # CPU-only node left out
        self.assertEqual((nodes["a5000-1"]["gpus_used"], nodes["rtxpro6000"]["gpus_used"]), (2, 4))
        self.assertEqual(nodes["rtxpro6000"]["restricted"], ["restricted-gpu=x"])
        kinds = {w["name"]: (w["kind"], w["state"]) for w in snap["workloads"]}
        self.assertEqual(kinds, {"nb": ("Notebook", "Running"), "train": ("kjob", "Running"), "big": ("Notebook", "Pending")})
        big = next(w for w in snap["workloads"] if w["name"] == "big")
        self.assertEqual(cluster.workload_type(big, nodes), "rtxpro6000")      # unlabeled node shown by host name
        self.assertIn("0/8 nodes", big["reason"])

    def test_falls_back_to_published_configmap(self):
        api = fake_api()
        api.list_node.side_effect = core.ApiException(status=403)
        api.read_namespaced_config_map.return_value = client.V1ConfigMap(data={"status.json": json.dumps({"nodes": [], "workloads": []})})
        self.assertEqual(cluster.load_snapshot(api)["source"], "published")
        api.read_namespaced_config_map.side_effect = core.ApiException(status=404)
        with self.assertRaises(cluster.ClusterViewUnavailable):
            cluster.load_snapshot(api)

    def test_type_matching_and_resolution(self):
        snap = cluster.build_snapshot(fake_api())
        self.assertEqual(cli.resolve_gpu_types(["a5000"], None, snap), (["a5000"], None))
        self.assertEqual(cli.resolve_gpu_types(["rtx-a5000"], None, snap), (["NVIDIA-RTX-A5000"], None))
        self.assertEqual(cli.resolve_gpu_types(["rtxpro6000"], None, snap), ([], "rtxpro6000"))
        self.assertEqual(cli.resolve_gpu_types(["RTX PRO 6000"], None, snap), ([], "rtxpro6000"))
        with self.assertRaises(ValueError):
            cli.resolve_gpu_types(["h100"], None, snap)

    def test_sinfo_and_queue_output(self):
        snap = {**cluster.build_snapshot(fake_api()), "source": "live"}
        out = io.StringIO()
        with mock.patch.object(cluster, "load_snapshot", return_value=snap), mock.patch.object(cli, "_my_tolerations", return_value=[]), \
                mock.patch.object(core, "current_namespace", return_value="bob"), mock.patch("sys.stdout", out):
            cli._sinfo(argparse.Namespace(gpu_type="rtxpro6000"))
        text = out.getvalue()
        self.assertIn("rtxpro6000", text)
        self.assertIn("no access", text)
        self.assertIn("8 (1 jobs)", text)          # carol waits for 8 GPUs on rtxpro6000
        self.assertIn("*bob", text)
        self.assertNotIn("alice", text)            # filtered to rtxpro6000


class TimeAndDirectiveTest(unittest.TestCase):
    def test_time_formats(self):
        for v, s in [("30", 1800), ("10:30", 630), ("2:00:00", 7200), ("1-12", 129600), ("1-00:30", 88200),
                     ("1-0:0:5", 86405), ("90m", 5400), ("12h", 43200), ("2d", 172800)]:
            self.assertEqual(cli.parse_time(v), s, v)
        with self.assertRaises(argparse.ArgumentTypeError):
            cli._time_arg("soon")

    def test_time_limit_in_manifest(self):
        m = manifest(time_limit=7200)
        self.assertEqual(m["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"]["activeDeadlineSeconds"], 7200)
        self.assertNotIn("activeDeadlineSeconds", m["spec"]["runPolicy"])     # would also count queueing time
        self.assertNotIn("activeDeadlineSeconds", manifest()["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"])

    def test_directives(self):
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
            f.write("#!/bin/bash\n#SBATCH -J train --cpus-per-task=8\n#SBATCH --mem=64G -t 1-00:00:00\n"
                    "#SBATCH --gres=gpu:v100:2 -p gp1d --account=ABC -N 1\n#KJOB --shm 16Gi\n\npython train.py\n#KJOB -G 7\n")
        err = io.StringIO()
        extra = cli.script_directives(f.name, warn=lambda m: err.write(m + "\n"))
        self.assertEqual(extra, ["-J", "train", "-c", "8", "--mem", "64Gi", "-t", "1-00:00:00", "-g", "v100", "-G", "2",
                                 "--shm", "16Gi"])                       # stops at the first command
        self.assertIn("-p", err.getvalue())
        self.assertIn("--account", err.getvalue())
        self.assertIn("-N 1", err.getvalue())
        self.assertNotIn("SBATCH 1", err.getvalue())
        a = cli.build_parser().parse_args(["submit"] + extra + ["-G", "1", f.name])
        self.assertEqual((a.gpus, a.cpus, a.time, a.job_name), (1, 8, 86400, "train"))   # command line wins
        Path(f.name).unlink()


class HistoryTest(unittest.TestCase):
    def test_history_and_log_result(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "j.log"
            log.write_text("[kjob 2026-01-01T00:00:00Z] start j on node n\nhello\n[kjob 2026-01-01T01:00:05Z] exit code 3\n")
            self.assertEqual(core.log_result(str(log)), ("2026-01-01T00:00:00Z", "2026-01-01T01:00:05Z", 3))
            job = mock.Mock(ns="ns")
            job.spec = core.JobSpec(name="j", script="x.py", log_dir=d)
            with mock.patch.object(core, "HISTORY_FILE", Path(d) / "h" / "history.jsonl"):
                core.record_history(job)
                self.assertEqual(core.read_history()[0]["name"], "j")
                out = io.StringIO()
                with mock.patch.object(core, "list_jobs", return_value=[]), mock.patch("sys.stdout", out):
                    cli.cmd_acct(argparse.Namespace(n=20))
            self.assertIn("FAILED", out.getvalue())
            self.assertIn("1h00m", out.getvalue())


class SkillsTest(unittest.TestCase):
    def test_install_skills(self):
        with tempfile.TemporaryDirectory() as d:
            cli.cmd_install_skills(argparse.Namespace(dir=d))
            names = sorted(p.parent.name for p in Path(d).glob("*/SKILL.md"))
        self.assertEqual(names, ["kjob-install", "kjob-jobs"])


class PublishTest(unittest.TestCase):
    def test_creates_configmap_when_missing_and_can_hide_names(self):
        api = fake_api()
        api.replace_namespaced_config_map.side_effect = core.ApiException(status=404)
        snap = cluster.build_snapshot(api, hide_names=True)
        cluster.publish(snap, api)
        body = api.create_namespaced_config_map.call_args[0][1]
        data = json.loads(body.data["status.json"])
        self.assertEqual({w["name"] for w in data["workloads"]}, {"-"})
        self.assertEqual({w["namespace"] for w in data["workloads"]}, {"alice", "bob", "carol"})
