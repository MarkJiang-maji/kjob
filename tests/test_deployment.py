"""Deployment regressions: real shell failures plus isolated Kubernetes lifecycle cases."""
import argparse
import io
import json
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kubernetes import client
from kjob import cli, cluster, core
from test_audit_fixes import fixture
from test_regressions import handle, pod, load_template, manifest_from
from test_kjob import pod_spec


class LogStorageTests(unittest.TestCase):
    def shell(self, directory, log, command):
        term = Path(directory) / 'termination.json'
        shell = core.job_shell('test', command, directory, str(log), 'abc123', str(term))
        p = subprocess.run(['bash', '-c', shell], capture_output=True, text=True, timeout=10,
                           env={**os.environ, 'TMPDIR': directory})
        return p, json.loads(term.read_text())

    def test_enospc_is_not_success_and_kubelet_gets_independent_result(self):
        for script_rc in (0, 7):
            with self.subTest(script_rc=script_rc), tempfile.TemporaryDirectory() as d:
                p, result = self.shell(d, '/dev/full', f"bash -c 'echo output; exit {script_rc}'")
                self.assertEqual(p.returncode, script_rc or 74)
                self.assertEqual(result, {'kjob': 1, 'run_id': 'abc123', 'log_saved': False,
                                          'script_exit_code': script_rc})
                self.assertIn('log write failed:', p.stdout)
                self.assertIn('output', p.stdout)

    def test_unwritable_log_never_starts_user_script(self):
        with tempfile.TemporaryDirectory() as d:
            blocked = Path(d) / 'blocked'
            blocked.write_text('file')
            marker = Path(d) / 'should-not-run'
            p, result = self.shell(d, blocked / 'test.log', 'touch ' + shlex.quote(str(marker)))
            self.assertEqual(p.returncode, 74)
            self.assertFalse(marker.exists())
            self.assertFalse(result['log_saved'])
            self.assertIsNone(result['script_exit_code'])

    def test_success_and_script_failure_keep_complete_unicode_output_with_spaces(self):
        for rc in (0, 9):
            with self.subTest(rc=rc), tempfile.TemporaryDirectory(prefix='kjob spaces ') as d:
                log = Path(d) / 'logs with spaces' / 'test.log'
                p, result = self.shell(d, log, f"bash -c 'printf \"測試結果\\n\"; exit {rc}'")
                self.assertEqual(p.returncode, rc)
                self.assertTrue(result['log_saved'])
                self.assertEqual(result['script_exit_code'], rc)
                self.assertIn('測試結果', log.read_text())
                self.assertEqual(core.log_result(str(log), 'abc123')[2], rc)


class MonitoringTests(unittest.TestCase):
    def test_newline_free_progress_does_not_buffer_the_entire_job_output(self):
        j = handle()
        j.uid = 'old'
        j.core.read_namespaced_pod.return_value = pod()
        text = '進度' * (core.LOG_FRAGMENT_CHARS + 17)
        raw = text.encode()
        # Chunks deliberately split UTF-8 characters; no replacement characters or data loss.
        j.core.read_namespaced_pod_log.return_value.stream.return_value = (
            raw[i:i+1024] for i in range(0, len(raw), 1024))
        fragments = list(j.stream_logs())
        self.assertEqual(''.join(fragments), text)
        self.assertLessEqual(max(map(len, fragments)), core.LOG_FRAGMENT_CHARS)
        j.core.read_namespaced_pod_log.return_value.close.assert_called_once()

    def finished(self, code=74, log_saved=False, state='Failed'):
        j = handle(gpus=0)
        j.uid, j.run_id = 'old', 'abc123'
        p = pod(phase='Failed' if code else 'Succeeded', code=code)
        p.status.container_statuses[0].state.terminated.message = json.dumps(
            {'kjob': 1, 'run_id': 'abc123', 'log_saved': log_saved, 'script_exit_code': 0})
        j.core.read_namespaced_pod.return_value = p
        j.wait_for_pod = mock.Mock(return_value=p.status.phase)
        j.state = mock.Mock(return_value=state)
        j.stream_logs = mock.Mock(return_value=iter(['[kjob now] exit code 0 run abc123']))
        j.submit, j.delete = mock.Mock(), mock.Mock()
        return j

    def test_ksrun_preserves_failed_log_pod_and_nonzero_exit(self):
        j, messages = self.finished(), []
        self.assertEqual(j.run(out=io.StringIO(), report=messages.append), 74)
        j.delete.assert_not_called()
        self.assertIn('keeping the finished job and Pod', messages[-1])
        self.assertIn('--ttl', messages[-1])
        self.assertEqual(core.read_events()[-1]['run_id'], 'abc123')

    def test_kwait_also_preserves_failed_log_pod(self):
        j = self.finished()
        with mock.patch.object(cli, '_job_for', return_value=j), mock.patch('sys.stderr', io.StringIO()):
            self.assertEqual(cli.cmd_wait(argparse.Namespace(name=j.spec.name, keep=False)), 74)
        j.delete.assert_not_called()

    def test_normal_completion_still_cleans_up(self):
        j = self.finished(code=0, log_saved=True)
        self.assertEqual(j.run(out=io.StringIO(), report=lambda _: None), 0)
        j.delete.assert_called_once()

    def test_main_exit_does_not_wait_for_a_stalled_operator(self):
        for code in (0, 7):
            j = self.finished(code=code, log_saved=True, state='Running')
            with self.subTest(code=code), mock.patch.object(core.time, 'sleep') as sleep:
                self.assertEqual(j.run(out=io.StringIO(), report=lambda _: None), code)
            sleep.assert_not_called()
            j.delete.assert_called_once()

    def test_kwait_cleans_up_when_main_finished_but_operator_still_running(self):
        j = self.finished(code=0, log_saved=True, state='Running')
        with mock.patch.object(cli, '_job_for', return_value=j), mock.patch('sys.stderr', io.StringIO()):
            self.assertEqual(cli.cmd_wait(argparse.Namespace(name=j.spec.name, keep=False)), 0)
        j.delete.assert_called_once()

    def test_unconfirmed_completion_returns_error_without_cancelling_job(self):
        j = self.finished(state='Running')
        j.core.read_namespaced_pod.return_value = pod()  # main has no termination status
        with mock.patch.object(core, 'STATUS_GRACE_SECONDS', 0):
            self.assertEqual(j.run(out=io.StringIO(), report=lambda _: None), 1)
        j.delete.assert_not_called()

    def test_wrong_run_metadata_is_not_used(self):
        j = self.finished()
        j.run_id = 'different'
        self.assertEqual(j.exit_code(), 74)
        self.assertFalse(getattr(j, 'log_failed', False))

    def test_log_error_accounting_is_bound_to_run_not_reused_name(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / 'same.log'
            log.write_text('[kjob 2026-01-01T00:00:01Z] start same on node n run aaa\n'
                           '[kjob 2026-01-01T00:00:02Z] exit code 0 run aaa\n'
                           '[kjob 2026-01-01T00:01:01Z] start same on node n run bbb\n'
                           '[kjob 2026-01-01T00:01:02Z] exit code 0 run bbb\n')
            history = [{'name': 'same', 'submitted': '2026-01-01T00:00:00Z', 'log': str(log), 'gpus': 0,
                        'run_id': rid} for rid in ('aaa', 'bbb')]
            events = [{'event': 'log-error', 'name': 'same', 'run_id': 'aaa', 'exit_code': 74}]
            out = io.StringIO()
            with mock.patch.object(core, 'read_history', return_value=history), \
                    mock.patch.object(core, 'read_events', return_value=events), \
                    mock.patch.object(core, 'list_jobs', return_value=[]), mock.patch('sys.stdout', out):
                cli.cmd_acct(argparse.Namespace(n=0))
            rows = out.getvalue().splitlines()[1:]
            self.assertEqual([row.split()[5:7] for row in rows], [['LOG_ERROR', '74'], ['COMPLETED', '0']])


class ResourceTests(unittest.TestCase):
    def test_restartable_init_overlap_and_order(self):
        def c(cpu, sidecar=False):
            return client.V1Container(name='c', resources=client.V1ResourceRequirements(requests={'cpu': str(cpu)}),
                                      restart_policy='Always' if sidecar else None)
        for init, expected in [([c(8, True)], 9.5), ([c(2, True), c(8), c(3, True)], 10.5),
                               ([c(8), c(2, True), c(3, True)], 8.5), ([c(8), c(9)], 9.5), ([], 1.5)]:
            p = client.V1Pod(spec=client.V1PodSpec(containers=[c(1)], init_containers=init, overhead={'cpu': '500m'}))
            with self.subTest(init=init):
                self.assertEqual(cluster._requests(p, 'cpu'), expected)

    def test_shm_option_does_not_resize_other_scratch_volumes(self):
        p = fixture()
        for name, medium, size in [('ram-scratch', 'Memory', '128Mi'), ('disk-scratch', '', '500Mi')]:
            p.spec.volumes.append(client.V1Volume(name=name, empty_dir=client.V1EmptyDirVolumeSource(
                medium=medium, size_limit=size)))
            p.spec.containers[0].volume_mounts.append(client.V1VolumeMount(name=name, mount_path='/'+name))
        t = load_template(p, {'items': []})
        m = pod_spec(manifest_from(t, shm='8Gi'))
        volumes = {v['name']: v for v in m['volumes']}
        self.assertEqual(volumes['ram-scratch']['emptyDir']['sizeLimit'], '128Mi')
        self.assertEqual(volumes['disk-scratch']['emptyDir']['sizeLimit'], '500Mi')
        self.assertEqual(volumes['shm']['emptyDir']['sizeLimit'], '8Gi')
