"""kjob side of the cluster's fair-scheduler: --priority, the Queued state and queue messages."""
import io
import unittest
from unittest import mock

from kjob import cli, core
from test_kjob import NS, manifest


def queued_job(position="2", reason="no node has enough free GPUs of the requested type", suspend=True):
    ann = {core.QUEUED_ANN: "2026-11-12T03:00:00Z"}
    if position:
        ann.update({core.POSITION_ANN: position, core.REASON_ANN: reason})
    return {"metadata": {"name": "train", "namespace": NS, "annotations": ann, "labels": {}},
            "spec": {"runPolicy": {"suspend": suspend}},
            "status": {"conditions": [{"type": "Created", "status": "True"},
                                      {"type": "Suspended", "status": "True" if suspend else "False"}]}}


class PriorityTest(unittest.TestCase):
    def test_label_defaults_to_normal(self):
        self.assertEqual(manifest()["metadata"]["labels"][core.PRIORITY_LABEL], "normal")
        self.assertEqual(manifest(priority="high")["metadata"]["labels"][core.PRIORITY_LABEL], "high")

    def test_unknown_priority_rejected(self):
        with self.assertRaises(ValueError):
            core.JobSpec(name="t", script="x", priority="xhigh").validate()

    def test_cli_option(self):
        p = cli.build_parser()
        self.assertEqual(p.parse_args(["submit", "--priority", "low", "x.py"]).priority, "low")
        self.assertEqual(p.parse_args(["submit", "x.py"]).priority, "normal")


class QueueStateTest(unittest.TestCase):
    def test_queued_until_released(self):
        self.assertEqual(core.job_state(queued_job()), "Queued")
        released = queued_job(suspend=False)
        released["status"]["conditions"].append({"type": "Running", "status": "True"})
        self.assertEqual(core.job_state(released), "Running")

    def test_suspended_without_scheduler_is_not_queued(self):
        job = queued_job()
        del job["metadata"]["annotations"][core.QUEUED_ANN]
        self.assertEqual(core.job_state(job), "Suspended")

    def test_terminal_wins(self):
        job = queued_job()
        job["status"]["conditions"].append({"type": "Failed", "status": "True"})
        self.assertEqual(core.job_state(job), "Failed")

    def test_notes(self):
        self.assertEqual(core.queue_note(queued_job()),
                         "queued (position 2): no node has enough free GPUs of the requested type")
        self.assertEqual(core.queue_note(queued_job(position=None)), "queued: waiting for the fair-scheduler")
        self.assertIn("queued by the fair-scheduler", core.submit_note(queued_job()))
        bypassed = {"metadata": {"annotations": {core.BYPASSED_ANN: "scheduler not running"}}, "spec": {}}
        self.assertIn("not running", core.submit_note(bypassed))
        self.assertEqual(core.submit_note({"metadata": {}, "spec": {}}), "")


class WaitTest(unittest.TestCase):
    def test_wait_reports_queue_position_once_per_change(self):
        states = [queued_job("3"), queued_job("3"), queued_job("1"), {"metadata": {"name": "train"}, "spec": {},
                                                                        "status": {"conditions": [
                                                                            {"type": "Failed", "status": "True"}]}}]
        with mock.patch.object(core, "_load_config"), mock.patch.object(core, "current_namespace", return_value=NS), \
                mock.patch.object(core.client, "CoreV1Api") as c, mock.patch.object(core.client, "CustomObjectsApi") as o, \
                mock.patch.object(core.time, "sleep"):
            for s in states:
                s["metadata"]["uid"] = "u1"
            o.return_value.get_namespaced_custom_object.side_effect = states
            c.return_value.read_namespaced_pod.side_effect = core.ApiException(status=404)
            c.return_value.list_namespaced_event.return_value.items = []
            j = cli._job_for("train")
            msgs = []
            self.assertEqual(j.wait_for_pod(report=msgs.append), "Failed")
        self.assertEqual([m for m in msgs if "queued" in m], [
            "kjob: train queued (position 3): no node has enough free GPUs of the requested type",
            "kjob: train queued (position 1): no node has enough free GPUs of the requested type"])


class SubmitOutputTest(unittest.TestCase):
    def test_submit_prints_queue_note(self):
        job = mock.Mock()
        job.submit.return_value = "train"
        job.spec.log_path.return_value = "/home/jovyan/kjob-logs/train.log"
        job.obj = queued_job()
        out = io.StringIO()
        with mock.patch.object(cli, "_spec"), mock.patch.object(core, "KJob", return_value=job), \
                mock.patch("sys.stdout", out):
            self.assertEqual(cli.cmd_submit(mock.Mock()), 0)
        self.assertIn("queued by the fair-scheduler", out.getvalue())


if __name__ == "__main__":
    unittest.main()
