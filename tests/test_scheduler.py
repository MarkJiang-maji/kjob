"""Fair scheduler: who is released next, and what one pass does against a mocked cluster."""
import datetime as dt
import unittest
from unittest import mock

from kubernetes import client
from kubernetes.client.rest import ApiException

from kjob import core, scheduler
from kjob.scheduler import GPU, Job, Node, pick_releases
from test_regressions import load_template, manifest_from


def job(ns, name, gpus=1, created="2026-11-04T01:00:00Z", **pod_spec):
    return Job(ns, name, created, {GPU: float(gpus)}, pod_spec)


def node(name="n1", gpus=4, **kw):
    return Node(name, {GPU: float(gpus)}, **kw)


def order(picks):
    return [f"{j.namespace}/{j.name}" for j, _ in picks]


class FairnessTests(unittest.TestCase):
    def test_submitting_more_jobs_gains_nothing(self):
        queued = [job("a", f"a{i}", created="2026-11-04T01:00:00Z") for i in range(100)]
        queued += [job("b", "b0", created="2026-11-04T02:00:00Z"), job("c", "c0", created="2026-11-04T03:00:00Z")]
        self.assertEqual(order(pick_releases(queued, [node(gpus=3)])), ["a/a0", "b/b0", "c/c0"])

    def test_the_namespace_holding_least_goes_first(self):
        queued = [job("busy", "old", created="2026-11-04T01:00:00Z"), job("idle", "new", created="2026-11-04T09:00:00Z")]
        self.assertEqual(order(pick_releases(queued, [node(gpus=1)], usage={"busy": 22})), ["idle/new"])

    def test_namespaces_alternate_as_capacity_is_handed_out(self):
        queued = [job("a", f"a{i}", created=f"2026-11-04T0{i}:00:00Z") for i in range(3)]
        queued += [job("b", f"b{i}", created=f"2026-11-04T0{i}:30:00Z") for i in range(3)]
        self.assertEqual(order(pick_releases(queued, [node(gpus=4)])), ["a/a0", "b/b0", "a/a1", "b/b1"])

    def test_free_capacity_is_not_held_back_when_others_have_nothing_queued(self):
        queued = [job("a", f"a{i}") for i in range(3)]
        self.assertEqual(len(pick_releases(queued, [node(gpus=3)], usage={"b": 0})), 3)

    def test_gpu_count_is_what_counts_not_job_count(self):
        queued = [job("big", "b1", gpus=3, created="2026-11-04T01:00:00Z"),
                  job("big", "b2", created="2026-11-04T02:00:00Z"),
                  job("small", "s1", created="2026-11-04T03:00:00Z"),
                  job("small", "s2", created="2026-11-04T04:00:00Z")]
        self.assertEqual(order(pick_releases(queued, [node(gpus=5)])), ["big/b1", "small/s1", "small/s2"])

    def test_a_job_that_does_not_fit_is_passed_over_within_its_namespace(self):
        queued = [job("a", "large", gpus=8, created="2026-11-04T01:00:00Z"), job("a", "small", created="2026-11-04T02:00:00Z")]
        self.assertEqual(order(pick_releases(queued, [node(gpus=4)])), ["a/small"])

    def test_nothing_released_without_capacity(self):
        self.assertEqual(pick_releases([job("a", "x")], [node(gpus=0)]), [])

    def test_a_large_job_cannot_be_split_across_nodes(self):
        self.assertEqual(pick_releases([job("a", "x", gpus=4)], [node("n1", 2), node("n2", 2)]), [])

    def test_tightest_node_is_used_so_large_gaps_survive(self):
        picks = pick_releases([job("a", "x")], [node("wide", 8), node("tight", 1)])
        self.assertEqual(picks[0][1], "tight")

    def test_the_callers_data_is_not_changed(self):
        nodes, usage, quota = [node(gpus=2)], {"a": 1}, {"a": {GPU: 5.0}}
        pick_releases([job("a", "x")], nodes, usage, quota)
        self.assertEqual((nodes[0].free[GPU], usage, quota), (2.0, {"a": 1}, {"a": {GPU: 5.0}}))


class LimitsTests(unittest.TestCase):
    def test_namespace_quota_blocks_only_that_namespace(self):
        queued = [job("full", "f", created="2026-11-04T01:00:00Z"), job("room", "r", created="2026-11-04T02:00:00Z")]
        self.assertEqual(order(pick_releases(queued, [node()], quota_left={"full": {GPU: 0.0}})), ["room/r"])

    def test_quota_is_spent_during_a_pass(self):
        queued = [job("a", "a0"), job("a", "a1")]
        self.assertEqual(len(pick_releases(queued, [node()], quota_left={"a": {GPU: 1.0}})), 1)

    def test_budget_caps_what_a_pass_hands_out(self):
        queued = [job("a", "a0"), job("b", "b0"), job("c", "c0")]
        self.assertEqual(len(pick_releases(queued, [node()], budget=2)), 2)


class PlacementTests(unittest.TestCase):
    AFFINITY = {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": [
        {"matchExpressions": [{"key": "nvidia.com/gpu.product", "operator": "In", "values": ["RTX-A5000"]}]}]}}}

    def test_gpu_type_constraint(self):
        nodes = [node("v100", labels={"nvidia.com/gpu.product": "V100"}), node("a5", labels={"nvidia.com/gpu.product": "RTX-A5000"})]
        self.assertEqual(pick_releases([job("a", "x", affinity=self.AFFINITY)], nodes)[0][1], "a5")

    def test_a_free_gpu_of_the_wrong_type_starts_nothing(self):
        nodes = [node("v100", labels={"nvidia.com/gpu.product": "V100"})]
        self.assertEqual(pick_releases([job("a", "x", affinity=self.AFFINITY)], nodes), [])

    def test_node_pinning_and_exclusion_by_name(self):
        pin = {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": [
            {"matchFields": [{"key": "metadata.name", "operator": "NotIn", "values": ["n1"]}]}]}}}
        self.assertEqual(pick_releases([job("a", "x", affinity=pin)], [node("n1"), node("n2")])[0][1], "n2")

    def test_taints_need_a_toleration(self):
        taint = {"key": "gpu", "value": "reserved", "effect": "NoSchedule"}
        self.assertEqual(pick_releases([job("a", "x")], [node(taints=[taint])]), [])
        tol = [{"key": "gpu", "operator": "Equal", "value": "reserved", "effect": "NoSchedule"}]
        self.assertEqual(len(pick_releases([job("a", "x", tolerations=tol)], [node(taints=[taint])])), 1)
        self.assertEqual(len(pick_releases([job("a", "x", tolerations=[{"operator": "Exists"}])], [node(taints=[taint])])), 1)

    def test_node_selector(self):
        nodes = [node("n1", labels={"disk": "hdd"}), node("n2", labels={"disk": "ssd"})]
        self.assertEqual(pick_releases([job("a", "x", nodeSelector={"disk": "ssd"})], nodes)[0][1], "n2")

    def test_cpu_and_memory_also_have_to_fit(self):
        j = Job("a", "x", "t", {GPU: 1.0, "cpu": 8.0})
        self.assertEqual(pick_releases([j], [Node("n1", {GPU: 4.0, "cpu": 2.0})]), [])


def job_obj(ns, name, gpus=1, suspend=True, status=None, annotations=None, fair=True, **meta):
    labels = {core.FAIR_LABEL: core.FAIR_QUEUE} if fair else {}
    return {"metadata": {"namespace": ns, "name": name, "creationTimestamp": meta.pop("created", "2026-11-04T01:00:00Z"),
                         "resourceVersion": "42", "labels": labels,
                         "annotations": {} if annotations is None else annotations, **meta},
            "spec": {"runPolicy": {"suspend": suspend},
                     "pytorchReplicaSpecs": {"Master": {"replicas": 1, "template": {"spec": {"containers": [
                         {"resources": {"requests": {GPU: str(gpus), "cpu": "1", "memory": "1Gi"}}}]}}}}},
            "status": status}


def api_node(name="n1", gpus=4):
    return client.V1Node(
        metadata=client.V1ObjectMeta(name=name, labels={}),
        spec=client.V1NodeSpec(),
        status=client.V1NodeStatus(allocatable={GPU: str(gpus), "cpu": "64", "memory": "256Gi"},
                                   conditions=[client.V1NodeCondition(type="Ready", status="True")]))


def api_pod(ns, name, gpus=1, node_name="n1"):
    c = client.V1Container(name="c", resources=client.V1ResourceRequirements(requests={GPU: str(gpus)}))
    return client.V1Pod(metadata=client.V1ObjectMeta(namespace=ns, name=name),
                        spec=client.V1PodSpec(containers=[c], node_name=node_name),
                        status=client.V1PodStatus(phase="Running" if node_name else "Pending"))


def api_quota(ns, hard, used):
    return client.V1ResourceQuota(metadata=client.V1ObjectMeta(namespace=ns, name="q"),
                                  status=client.V1ResourceQuotaStatus(hard=hard, used=used))


class ReconcileTests(unittest.TestCase):
    def apis(self, jobs, pods=(), nodes=None, quotas=()):
        custom, api = mock.Mock(), mock.Mock()
        custom.list_cluster_custom_object.return_value = {"items": list(jobs)}
        api.list_pod_for_all_namespaces.return_value.items = list(pods)
        api.list_node.return_value.items = [api_node()] if nodes is None else nodes
        api.list_resource_quota_for_all_namespaces.return_value.items = list(quotas)
        return custom, api

    def run_pass(self, custom, api, **kw):
        lines = []
        picks = scheduler.reconcile(custom, api, report=lines.append, **kw)
        return [j.name for j, _ in picks], lines

    def test_only_fair_jobs_are_asked_for(self):
        custom, api = self.apis([])
        self.run_pass(custom, api)
        self.assertEqual(custom.list_cluster_custom_object.call_args.kwargs["label_selector"], "kjob/queue=fair")

    def test_no_cluster_reads_when_nothing_is_queued(self):
        custom, api = self.apis([])
        self.run_pass(custom, api)
        api.list_pod_for_all_namespaces.assert_not_called()

    def test_release_removes_suspend_and_records_when(self):
        custom, api = self.apis([job_obj("a", "x")])
        self.assertEqual(self.run_pass(custom, api)[0], ["x"])
        args = custom.patch_namespaced_custom_object.call_args.args
        self.assertEqual(args[2:5], ("a", "pytorchjobs", "x"))
        patch = args[5]
        self.assertIsInstance(patch, dict)     # the API server rejected a JSON patch list: clients send merge-patch
        self.assertIs(patch["spec"]["runPolicy"]["suspend"], False)
        self.assertEqual(patch["metadata"]["resourceVersion"], "42")
        self.assertIn(scheduler.RELEASED, patch["metadata"]["annotations"])

    def test_re_suspended_job_is_never_restarted(self):
        custom, api = self.apis([job_obj("a", "x", status={"startTime": "2026-11-04T01:00:00Z"})])
        self.assertEqual(self.run_pass(custom, api)[0], [])
        custom.patch_namespaced_custom_object.assert_not_called()

    def test_job_being_deleted_is_left_alone(self):
        custom, api = self.apis([job_obj("a", "x", deletionTimestamp="2026-11-04T02:00:00Z")])
        self.assertEqual(self.run_pass(custom, api)[0], [])

    def test_running_jobs_are_never_patched(self):
        custom, api = self.apis([job_obj("a", "run", suspend=False), job_obj("a", "x")],
                                pods=[api_pod("a", "run-master-0")])
        self.run_pass(custom, api)
        self.assertEqual([c.args[4] for c in custom.patch_namespaced_custom_object.call_args_list], ["x"])

    def test_notebooks_count_as_usage_and_take_capacity(self):
        jobs = [job_obj("nb", "older", created="2026-11-04T01:00:00Z"), job_obj("free", "newer", created="2026-11-04T02:00:00Z")]
        custom, api = self.apis(jobs, pods=[api_pod("nb", "notebook-0", gpus=3)])
        self.assertEqual(self.run_pass(custom, api)[0], ["newer"])

    def test_a_stuck_unbound_notebook_holds_nothing(self):
        custom, api = self.apis([job_obj("a", "x")], pods=[api_pod("b", "stuck-0", gpus=4, node_name=None)])
        self.assertEqual(self.run_pass(custom, api)[0], ["x"])

    def test_released_job_without_a_pod_keeps_its_gpu_reserved(self):
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        released = job_obj("a", "released", gpus=4, suspend=False, annotations={scheduler.RELEASED: stamp})
        custom, api = self.apis([released, job_obj("b", "next")])
        self.assertEqual(self.run_pass(custom, api)[0], [])

    def test_released_job_stops_reserving_after_the_timeout(self):
        released = job_obj("a", "stuck", gpus=4, suspend=False, annotations={scheduler.RELEASED: "2020-01-01T00:00:00Z"})
        custom, api = self.apis([released, job_obj("b", "next")])
        self.assertEqual(self.run_pass(custom, api)[0], ["next"])

    def test_bad_release_stamp_is_not_trusted(self):
        released = job_obj("a", "odd", gpus=4, suspend=False, annotations={scheduler.RELEASED: "soon"})
        custom, api = self.apis([released, job_obj("b", "next")])
        self.assertEqual(self.run_pass(custom, api)[0], ["next"])

    def test_bound_pod_is_not_counted_twice(self):
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        running = job_obj("a", "run", gpus=3, suspend=False, annotations={scheduler.RELEASED: stamp})
        custom, api = self.apis([running, job_obj("b", "next")], pods=[api_pod("a", "run-master-0", gpus=3)])
        self.assertEqual(self.run_pass(custom, api)[0], ["next"])

    def test_finished_jobs_hold_nothing(self):
        done = job_obj("a", "done", gpus=4, suspend=False, status={"conditions": [{"type": "Succeeded", "status": "True"}]},
                       annotations={scheduler.RELEASED: "2099-01-01T00:00:00Z"})
        custom, api = self.apis([done, job_obj("b", "next")])
        self.assertEqual(self.run_pass(custom, api)[0], ["next"])

    def test_quota_from_the_cluster_is_respected(self):
        quota = api_quota("a", {"requests.nvidia.com/gpu": "2"}, {"requests.nvidia.com/gpu": "2"})
        custom, api = self.apis([job_obj("a", "x")], quotas=[quota])
        self.assertEqual(self.run_pass(custom, api)[0], [])

    def test_max_in_use_counts_jobs_already_released(self):
        running = job_obj("a", "run", suspend=False)
        custom, api = self.apis([running, job_obj("a", "x"), job_obj("b", "y")], pods=[api_pod("a", "run-master-0")])
        self.assertEqual(self.run_pass(custom, api, max_in_use=2)[0], ["y"])

    def test_namespaces_limit_what_is_managed(self):
        custom, api = self.apis([job_obj("a", "x"), job_obj("b", "y")])
        self.assertEqual(self.run_pass(custom, api, namespaces=["b"])[0], ["y"])

    def test_dry_run_changes_nothing(self):
        custom, api = self.apis([job_obj("a", "x")])
        names, lines = self.run_pass(custom, api, dry_run=True)
        self.assertEqual(names, ["x"])
        custom.patch_namespaced_custom_object.assert_not_called()
        self.assertIn("would release a/x", lines[0])

    def test_one_failed_release_does_not_stop_the_rest(self):
        custom, api = self.apis([job_obj("a", "gone"), job_obj("b", "next")])
        custom.patch_namespaced_custom_object.side_effect = [ApiException(status=404, reason="Not Found"), {}]
        names, lines = self.run_pass(custom, api)
        self.assertEqual(names, ["next"])
        self.assertTrue(any("could not release a/gone: 404" in l for l in lines))

    def test_shapes_it_cannot_place_are_reported_not_released(self):
        obj = job_obj("a", "dist")
        obj["spec"]["pytorchReplicaSpecs"]["Worker"] = {"replicas": 2}
        custom, api = self.apis([obj])
        names, lines = self.run_pass(custom, api)
        self.assertEqual(names, [])
        self.assertIn("not a single Master pod", lines[0])

    def test_unready_and_cordoned_nodes_offer_nothing(self):
        cordoned = api_node()
        cordoned.spec.unschedulable = True
        custom, api = self.apis([job_obj("a", "x")], nodes=[cordoned])
        self.assertEqual(self.run_pass(custom, api)[0], [])
        unready = api_node()
        unready.status.conditions[0].status = "False"
        custom, api = self.apis([job_obj("a", "x")], nodes=[unready])
        self.assertEqual(self.run_pass(custom, api)[0], [])


class JobStateTests(unittest.TestCase):
    def test_queued_is_not_reported_as_pending(self):
        self.assertEqual(core.job_state(job_obj("a", "x")), "Queued")

    def test_re_suspended_job_is_not_reported_as_running(self):
        stale = {"startTime": "2026-11-04T01:00:00Z", "conditions": [{"type": "Running", "status": "True"}]}
        self.assertEqual(core.job_state(job_obj("a", "x", status=stale)), "Suspended")

    def test_suspended_job_outside_the_fair_queue_is_not_queued(self):
        self.assertEqual(core.job_state(job_obj("a", "x", fair=False)), "Suspended")

    def test_finished_stays_finished_even_if_suspended(self):
        done = {"conditions": [{"type": "Succeeded", "status": "True"}]}
        self.assertEqual(core.job_state(job_obj("a", "x", status=done)), "Succeeded")

    def test_unsuspended_jobs_keep_their_states(self):
        self.assertEqual(core.job_state(job_obj("a", "x", suspend=False)), "Pending")
        running = {"conditions": [{"type": "Running", "status": "True"}]}
        self.assertEqual(core.job_state(job_obj("a", "x", suspend=False, status=running)), "Running")


class SubmissionTests(unittest.TestCase):
    def test_fair_job_is_created_suspended_and_labelled(self):
        m = manifest_from(load_template(), fair=True)
        self.assertIs(m["spec"]["runPolicy"]["suspend"], True)
        self.assertEqual(m["metadata"]["labels"][core.FAIR_LABEL], core.FAIR_QUEUE)

    def test_default_submission_is_unchanged(self):
        m = manifest_from(load_template())
        self.assertNotIn("suspend", m["spec"]["runPolicy"])
        self.assertNotIn(core.FAIR_LABEL, m["metadata"]["labels"])

    def test_waiting_for_a_queued_job_says_why(self):
        j = core.KJob.__new__(core.KJob)
        j.spec = core.JobSpec(name="x", script="")
        j.ns, j.pod_name = "a", "x-master-0"
        j.core, j.custom = mock.Mock(), mock.Mock()
        j.core.read_namespaced_pod.side_effect = ApiException(status=404)
        lines = []
        with mock.patch.object(core, "job_events", return_value=[]), mock.patch.object(core.time, "sleep"), \
                mock.patch.object(j, "state", side_effect=["Queued", "Queued", "Deleted"]):
            self.assertEqual(j.wait_for_pod(report=lines.append), "Deleted")
        self.assertEqual(lines, ["kjob: x queued: waiting for the fair scheduler to release it"])


if __name__ == "__main__":
    unittest.main()
