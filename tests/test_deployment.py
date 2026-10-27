"""Deployment regressions: real shell failures plus isolated Kubernetes lifecycle cases."""
import argparse
import io
import json
import os
import shlex
import subprocess
import tempfile
import queue
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from kubernetes import client
import urllib3
from kjob import cli, cluster, core
from test_audit_fixes import fixture
from test_regressions import handle, pod, load_template, manifest_from
from test_kjob import pod_spec


class LogStorageTests(unittest.TestCase):
    def test_progress_is_line_framed_for_the_runtime_but_saved_bytes_are_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            log, term = Path(d)/'job.log', Path(d)/'termination.json'
            command = "bash -c 'printf \"\\r10%%\\r20%%\\r\\ncomplete\\n\"'"
            shell = core.job_shell('test', command, d, str(log), 'abc123', str(term))
            p = subprocess.run(['bash','-c',shell], capture_output=True, timeout=10,
                               env={**os.environ,'TMPDIR':d})
            self.assertEqual(p.returncode, 0)
            self.assertIn(b'\r10%\r20%\r\ncomplete\n', log.read_bytes())
            self.assertIn(b'\r\n10%\r\n20%\r\ncomplete\n', p.stdout)

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
    def test_http_progress_is_visible_while_the_connection_is_still_open(self):
        for chunked in (True, False, 'slow-chunk'):
            with self.subTest(chunked=chunked):
                release = threading.Event()
                class Handler(BaseHTTPRequestHandler):
                    protocol_version = 'HTTP/1.1'
                    def log_message(self, *args):
                        pass
                    def do_GET(self):
                        self.send_response(200)
                        self.send_header('Transfer-Encoding' if chunked else 'Connection',
                                         'chunked' if chunked else 'close')
                        self.end_headers()
                        self.wfile.write(b'8\r\n1%\r' if chunked == 'slow-chunk'
                                         else b'3\r\n1%\r\r\n' if chunked else b'1%\r')
                        self.wfile.flush()
                        release.wait(5)
                        self.wfile.write(b'done\n\r\n0\r\n\r\n' if chunked == 'slow-chunk'
                                         else b'5\r\ndone\n\r\n0\r\n\r\n' if chunked else b'done\n')
                        self.wfile.flush()
                        self.close_connection = True
                server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
                serving = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.05), daemon=True)
                serving.start()
                pool = urllib3.HTTPConnectionPool('127.0.0.1', server.server_port, timeout=3)
                response = pool.request('GET', '/', preload_content=False)
                j = handle()
                j.core.read_namespaced_pod.return_value = pod()
                j.core.read_namespaced_pod_log.return_value = response
                received = queue.Queue()
                def consume():
                    try:
                        for update in j.stream_logs():
                            received.put(update)
                    except Exception as error:
                        received.put(error)
                consumer = threading.Thread(target=consume, daemon=True)
                consumer.start()
                try:
                    self.assertEqual(received.get(timeout=2), '1%\r')
                    self.assertFalse(release.is_set())
                finally:
                    release.set()
                    consumer.join(4)
                    server.shutdown()
                    server.server_close()
                    pool.close()
                    serving.join(1)
                self.assertFalse(consumer.is_alive())
                tail = []
                while not received.empty():
                    tail.append(received.get_nowait())
                self.assertTrue(all(isinstance(x, str) for x in tail), tail)
                self.assertEqual(''.join(x if isinstance(x, core.LogFragment) else x+'\n' for x in tail), 'done\n')

    def test_short_carriage_return_updates_arrive_before_the_stream_ends(self):
        j = handle()
        j.core.read_namespaced_pod.return_value = pod()
        response = j.core.read_namespaced_pod_log.return_value
        def chunks():
            yield '進度 1\r'.encode()
            self.assertEqual(output.getvalue(), '進度 1\r')
            yield b'progress 2\r'
            self.assertEqual(output.getvalue(), '進度 1\rprogress 2\r')
            yield b'done\n'
        response.stream.return_value = chunks()
        j.wait_for_pod = mock.Mock(return_value='Running')
        j.state = mock.Mock(return_value='Succeeded')
        j.exit_code = mock.Mock(return_value=0)
        j.stop_reason = mock.Mock(return_value=None)
        output = io.StringIO()
        self.assertEqual(j.follow(out=output, report=lambda _: None), 0)
        self.assertEqual(output.getvalue(), '進度 1\rprogress 2\rdone\n')

    def test_tqdm_leading_carriage_return_does_not_wait_for_the_next_update(self):
        j = self.finished(code=0, log_saved=True)
        del j.stream_logs  # exercise the real reader and follow renderer
        output = io.StringIO()
        def chunks():
            yield b'\r10%'
            self.assertEqual(output.getvalue(), '\r10%')
            yield b'\r20'
            self.assertEqual(output.getvalue(), '\r10%\r20')
            yield b'%\n'
        j.core.read_namespaced_pod_log.return_value.stream.return_value = chunks()
        self.assertEqual(j.follow(out=output, report=lambda _: None), 0)
        self.assertEqual(output.getvalue(), '\r10%\r20%\n')

    def test_klogs_follow_preserves_carriage_returns(self):
        j = handle()
        j.state = mock.Mock(return_value='Running')
        j.wait_for_pod = mock.Mock(return_value='Running')
        j.stream_logs = mock.Mock(return_value=iter(['10%\r', '100%\r', '']))
        output = io.StringIO()
        with mock.patch.object(cli, '_job_for', return_value=j), mock.patch('sys.stdout', output):
            self.assertEqual(cli.cmd_logs(argparse.Namespace(name=j.spec.name, follow=True)), 0)
        self.assertEqual(output.getvalue(), '10%\r100%\r\n')

    def test_crlf_and_split_utf8_remain_intact(self):
        j = handle()
        j.core.read_namespaced_pod.return_value = pod()
        raw = '測試\r\nprogress\rfinal\n'.encode()
        j.core.read_namespaced_pod_log.return_value.stream.return_value = (bytes([b]) for b in raw)
        updates = list(j.stream_logs())
        text = ''.join(x if isinstance(x, core.LogFragment) or x.endswith('\r') else x+'\n' for x in updates)
        self.assertEqual(text, raw.decode())

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
            messages = []
            with self.subTest(code=code), mock.patch.object(core.time, 'sleep') as sleep:
                self.assertEqual(j.run(out=io.StringIO(), report=messages.append), code)
            sleep.assert_not_called()
            self.assertFalse(any('operator status' in message for message in messages))
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

    def test_retry_follow_does_not_keep_an_old_monitoring_error(self):
        j = self.finished(state='Running')
        p = pod()
        j.core.read_namespaced_pod.return_value = p
        with mock.patch.object(core, 'STATUS_GRACE_SECONDS', 0):
            self.assertEqual(j.follow(out=io.StringIO(), report=lambda _: None), 1)
        p.status.container_statuses = self.finished(code=0, log_saved=True).core.read_namespaced_pod.return_value.status.container_statuses
        j.state.return_value = 'Succeeded'
        self.assertEqual(j.follow(out=io.StringIO(), report=lambda _: None), 0)
        self.assertTrue(j.cleanup_finished(report=lambda _: None))
        j.delete.assert_called_once()

    def test_wrong_run_metadata_is_not_used(self):
        j = self.finished()
        j.run_id = 'different'
        self.assertEqual(j.exit_code(), 74)
        self.assertFalse(getattr(j, 'log_failed', False))

    def test_missing_container_diagnostic_goes_to_stderr_only_after_eviction(self):
        diagnostic = 'rpc error: code = Unknown desc = Error response from daemon: No such container: example'
        for reason in ('Evicted', None):
            with self.subTest(reason=reason):
                j = self.finished(code=143, log_saved=True)
                j.core.read_namespaced_pod.return_value.status.reason = reason
                j.stream_logs = mock.Mock(return_value=iter(['saved checkpoint', diagnostic]))
                out, messages = io.StringIO(), []
                self.assertEqual(j.follow(out=out, report=messages.append), 143)
                self.assertIn('saved checkpoint', out.getvalue())
                if reason:
                    self.assertNotIn('rpc error:', out.getvalue())
                    self.assertTrue(any('log stream unavailable after eviction' in m for m in messages))
                    self.assertIn('Evicted', messages[-1])
                else:
                    self.assertIn(diagnostic, out.getvalue())

    def test_klogs_keeps_eviction_diagnostics_separate_from_output(self):
        diagnostic = 'rpc error: code = Unknown desc = Error response from daemon: No such container: example'
        for follow in (True, False):
            with self.subTest(follow=follow):
                j = self.finished(code=143, log_saved=True)
                j.core.read_namespaced_pod.return_value.status.reason = 'Evicted'
                j.stream_logs = mock.Mock(return_value=iter(['checkpoint saved', diagnostic]))
                j.core.read_namespaced_pod_log.return_value = 'checkpoint saved\n'+diagnostic+'\n'
                out, err = io.StringIO(), io.StringIO()
                with mock.patch.object(cli, '_job_for', return_value=j), \
                        mock.patch('sys.stdout', out), mock.patch('sys.stderr', err):
                    self.assertEqual(cli.cmd_logs(argparse.Namespace(name=j.spec.name, follow=follow)), 0)
                self.assertEqual(out.getvalue(), 'checkpoint saved\n')
                self.assertIn(diagnostic, err.getvalue())

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

    def accounting_result(self, j, state='Running'):
        obj = {'metadata': {'name': j.spec.name, 'uid': 'old', 'annotations': {'kjob/run-id': 'abc123'}}}
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / 'job.log'
            log.write_text('[kjob 2026-01-01T00:00:01Z] start job on node n run abc123\n'
                           '[kjob 2026-01-01T00:00:02Z] exit code 0 run abc123\n')
            hist = [{'name': j.spec.name, 'run_id': 'abc123', 'submitted': '2026-01-01T00:00:00Z',
                     'log': str(log), 'gpus': 0}]
            out = io.StringIO()
            with mock.patch.object(core, 'read_history', return_value=hist), \
                    mock.patch.object(core, 'read_events', return_value=[]), \
                    mock.patch.object(core, 'list_jobs', return_value=[obj]), \
                    mock.patch.object(core, 'job_state', return_value=state), \
                    mock.patch.object(cli, '_job_for', return_value=j), mock.patch('sys.stdout', out):
                cli.cmd_acct(argparse.Namespace(n=0))
            return out.getvalue().splitlines()[1].split()[5:7]

    def test_background_accounting_checks_wrapper_status_without_kwait(self):
        for code, saved, expected in [(74, False, ['LOG_ERROR', '74']), (0, True, ['COMPLETED', '0'])]:
            with self.subTest(code=code):
                self.assertEqual(self.accounting_result(self.finished(code=code, log_saved=saved)), expected)

    def test_background_accounting_does_not_call_log_exit_a_finished_container(self):
        for state, expected in [('Created', 'PENDING'), ('Running', 'RUNNING'), ('Succeeded', 'UNCONFIRMED')]:
            j = self.finished()
            j.core.read_namespaced_pod.return_value = pod()  # no terminated main container yet
            with self.subTest(state=state):
                self.assertEqual(self.accounting_result(j, state), [expected, '-'])

    def test_background_accounting_api_failure_does_not_report_success(self):
        j = self.finished()
        j.core.read_namespaced_pod.side_effect = core.ApiException(status=503)
        self.assertEqual(self.accounting_result(j), ['UNCONFIRMED', '-'])

    def test_background_accounting_does_not_read_a_replacement_pod_result(self):
        j = self.finished()
        j.core.read_namespaced_pod.return_value = pod(uid='new', phase='Succeeded', code=0)
        self.assertEqual(self.accounting_result(j), ['RUNNING', '-'])  # CR state only; foreign exit 0 is ignored


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
