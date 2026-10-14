"""Regression tests for submission failures and job/Pod replacement races."""
import argparse
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from kubernetes import client
from kjob import cli, core
from test_kjob import notebook_pod, PODDEFAULTS, PROC_NFS, template, manifest, pod_spec


def load_template(pod=None, defaults=None, mounts=None):
    c, o = mock.Mock(), mock.Mock()
    c.read_namespaced_pod.return_value = pod or notebook_pod()
    o.list_namespaced_custom_object.return_value = defaults if defaults is not None else PODDEFAULTS
    with mock.patch.object(core, '_load_config'), mock.patch.object(core, 'current_namespace', return_value='mtest'), \
         mock.patch.object(core, 'current_notebook', return_value='notebook-0'), \
         mock.patch.object(core, '_nfs_mounts', return_value=mounts if mounts is not None else PROC_NFS), \
         mock.patch.object(core.os, 'getuid', return_value=0), \
         mock.patch.object(core.client, 'CoreV1Api', return_value=c), \
         mock.patch.object(core.client, 'CustomObjectsApi', return_value=o):
        return core.NotebookTemplate.from_current_pod()


def handle(**kw):
    with mock.patch.object(core.client, 'CoreV1Api'), mock.patch.object(core.client, 'CustomObjectsApi'):
        j = core.KJob(core.JobSpec(name='review-job', script='/home/jovyan/train.py', **kw), template())
    return j


def pod(uid='old', phase='Running', code=None):
    return client.V1Pod(metadata=client.V1ObjectMeta(owner_references=[client.V1OwnerReference(
        api_version='kubeflow.org/v1', kind='PyTorchJob', name='review-job', uid=uid)]),
        spec=client.V1PodSpec(containers=[], node_name='gpu-node'), status=client.V1PodStatus(
            phase=phase, container_statuses=[] if code is None else [client.V1ContainerStatus(
                name='pytorch', image='test', image_id='test', ready=False, restart_count=0,
                state=client.V1ContainerState(terminated=client.V1ContainerStateTerminated(exit_code=code)))]))


class StorageTests(unittest.TestCase):
    def test_readonly_pvc_and_mount_stay_readonly(self):
        for source_only in (False, True):
            p = notebook_pod()
            p.spec.volumes[0].persistent_volume_claim.read_only = source_only
            p.spec.containers[0].volume_mounts[0].read_only = not source_only
            t = load_template(p)
            self.assertTrue(t.volumes[0].nfs.read_only)
            self.assertTrue(t.volume_mounts[0].read_only)

    def test_nfs_subpath_is_rejected_before_wrong_export_is_mounted(self):
        p = notebook_pod()
        p.spec.containers[0].volume_mounts[0].sub_path = 'project'
        with self.assertRaisesRegex(ValueError, 'subPath'):
            load_template(p)

    def test_nonmatching_poddefault_does_not_remove_dataset(self):
        pd = {'items': [{'spec': {'selector': {'matchLabels': {'dataset': 'enabled'}},
                                  'volumes': [{'name': 'datasets'}]}}]}
        t = load_template(defaults=pd)
        self.assertIn('datasets', [v.name for v in t.volumes])

    def test_matching_selector_is_carried_to_job(self):
        p = notebook_pod()
        p.metadata.labels = {'dataset': 'enabled', 'unrelated': 'do-not-copy'}
        pd = {'items': [{'spec': {'selector': {'matchLabels': {'dataset': 'enabled'}},
                                  'volumes': [{'name': 'datasets'}]}}]}
        t = load_template(p, pd)
        self.assertNotIn('datasets', [v.name for v in t.volumes])
        with mock.patch.object(core.client, 'CoreV1Api'), mock.patch.object(core.client, 'CustomObjectsApi'):
            m = core.KJob(core.JobSpec(name='t', script='train.py'), t).manifest()
        self.assertEqual(m['spec']['pytorchReplicaSpecs']['Master']['template']['metadata']['labels']['dataset'], 'enabled')
        self.assertNotIn('unrelated', t.job_labels)

    def test_container_identity_is_preserved_without_extra_capabilities(self):
        p = notebook_pod()
        p.spec.containers[0].security_context = client.V1SecurityContext(
            run_as_user=1000, run_as_group=1000, run_as_non_root=True, privileged=True)
        t = load_template(p)
        self.assertEqual(t.effective_uid, 1000)
        with mock.patch.object(core.client, 'CoreV1Api'), mock.patch.object(core.client, 'CustomObjectsApi'):
            m = pod_spec(core.KJob(core.JobSpec(name='t', script='train.py'), t).manifest())
        self.assertEqual(m['containers'][0]['securityContext'], {'runAsUser': 1000, 'runAsGroup': 1000, 'runAsNonRoot': True})

    def test_nonroot_mps_is_rejected_instead_of_changing_uid(self):
        t = template()
        t.effective_uid = 1000
        with mock.patch.object(core.client, 'CoreV1Api'), mock.patch.object(core.client, 'CustomObjectsApi'):
            j = core.KJob(core.JobSpec(name='t', script='train.py', gpu_types=['a30-shared']), t)
        with self.assertRaisesRegex(ValueError, 'UID 0'):
            j.manifest()

    def test_proc_mounts_decodes_spaces(self):
        with mock.patch.object(core.Path, 'read_text', return_value='server:/export\\040space /mnt/a\\040b nfs4 rw 0 0\n'):
            self.assertEqual(core._nfs_mounts(), {'/mnt/a b': ('server', '/export space')})

    def test_script_on_emptydir_shadow_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            script = Path(root, 'tmp', 'train.sh')
            script.parent.mkdir()
            script.write_text('echo ok')
            j = handle(gpus=0)
            j.spec.script = str(script)
            j.spec.workdir = root
            j.spec.log_dir = root
            j.t.volumes = [client.V1Volume(name='home', nfs=client.V1NFSVolumeSource(server='s', path='/p')),
                           client.V1Volume(name='tmp', empty_dir=client.V1EmptyDirVolumeSource())]
            j.t.volume_mounts = [client.V1VolumeMount(name='home', mount_path=root),
                                client.V1VolumeMount(name='tmp', mount_path=str(script.parent))]
            with self.assertRaisesRegex(ValueError, 'not on a shared volume'):
                j._warn_unshared_paths()


class SubmissionTests(unittest.TestCase):
    def test_default_request_fits_teaching_quota(self):
        s = core.JobSpec(name='t', script='train.py')
        self.assertEqual((s.cpus, s.memory, s.shm), (1, '2Gi', '256Mi'))

    def test_invalid_resources_and_mixed_sharing_fail_early(self):
        for values in ({'gpus': -1}, {'cpus': 0}, {'ttl_seconds': -1}, {'memory': 'nan'}, {'memory': 'bogus'},
                       {'shm': '0'}, {'gpus': 2, 'gpu_types': ['a30-shared']},
                       {'gpu_types': ['a30', 'a30-shared']}, {'gpus': 0, 'gpu_types': ['a30']}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                core.JobSpec(name='t', script='train.py', **values).validate()

    def test_reserved_gpu_runtime_environment_is_rejected(self):
        for key in ('NVIDIA_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'CUDA_MPS_PIPE_DIRECTORY'):
            with self.assertRaises(ValueError):
                core.JobSpec(name='t', script='train.py', gpus=0, env={key: 'all'}).validate()

    def test_cpu_user_environment_is_supported(self):
        m = pod_spec(manifest(gpus=0, env={'EXPERIMENT': '42'}))
        self.assertIn({'name': 'EXPERIMENT', 'value': '42'}, m['containers'][0]['env'])

    def test_shared_request_has_mps_affinity(self):
        m = pod_spec(manifest(gpu_types=['a30-shared']))
        terms = m['affinity']['nodeAffinity']['requiredDuringSchedulingIgnoredDuringExecution']['nodeSelectorTerms'][0]['matchExpressions']
        self.assertIn({'key': core.SHARING_STRATEGY_LABEL, 'operator': 'In', 'values': ['mps']}, terms)

    def test_pinned_node_permission_failure_falls_back_to_published_status(self):
        j = handle(node='mps-node')
        j.core.read_node.side_effect = core.ApiException(status=403)
        snap = {'nodes': [{'name': 'mps-node', 'product': 'NVIDIA-A30-SHARED', 'sharing': 'mps'}]}
        with mock.patch.object(core.cluster, 'load_snapshot', return_value=snap):
            self.assertEqual(j._node_labels('mps-node')[core.SHARING_STRATEGY_LABEL], 'mps')
        with mock.patch.object(core.cluster, 'load_snapshot', side_effect=core.cluster.ClusterViewUnavailable('x')), \
                mock.patch('sys.stderr', new_callable=io.StringIO) as err:
            self.assertEqual(j._node_labels('mps-node'), {})
        self.assertIn('whole GPUs', err.getvalue())

    def test_authorized_pinned_mps_node_removes_shm(self):
        j = handle(node='mps-node')
        j.core.read_node.return_value = client.V1Node(metadata=client.V1ObjectMeta(labels={
            core.SHARING_STRATEGY_LABEL: 'mps', 'nvidia.com/gpu.product': 'NVIDIA-A30-SHARED'}))
        j.core.list_namespaced_resource_quota.return_value.items = []
        j.custom.create_namespaced_custom_object.return_value = {'metadata': {'uid': 'new'}}
        with mock.patch.object(j, '_warn_unshared_paths'):
            j.submit()
        m = pod_spec(j.custom.create_namespaced_custom_object.call_args.args[-1])
        self.assertNotIn('/dev/shm', [vm['mountPath'] for vm in m['containers'][0]['volumeMounts']])
        self.assertEqual(j.uid, 'new')

    def test_impossible_quota_is_rejected_before_post(self):
        j = handle(memory='32Gi')
        j.core.list_namespaced_resource_quota.return_value.items = [client.V1ResourceQuota(
            metadata=client.V1ObjectMeta(name='q'), status=client.V1ResourceQuotaStatus(hard={'requests.memory': '18000Mi'}))]
        with self.assertRaisesRegex(ValueError, 'waiting cannot satisfy'):
            j.submit()
        j.custom.create_namespaced_custom_object.assert_not_called()

    def test_temporarily_full_quota_still_allows_operator_retry(self):
        j = handle()
        j.core.list_namespaced_resource_quota.return_value.items = [client.V1ResourceQuota(
            metadata=client.V1ObjectMeta(name='q'), status=client.V1ResourceQuotaStatus(
                hard={'requests.nvidia.com/gpu': '1'}, used={'requests.nvidia.com/gpu': '1'}))]
        j.custom.create_namespaced_custom_object.return_value = {'metadata': {'uid': 'new'}}
        with mock.patch.object(j, '_warn_unshared_paths'):
            j.submit()
        j.custom.create_namespaced_custom_object.assert_called_once()

    def test_auto_names_are_unique_and_keep_suffix_for_long_stems(self):
        names = {core.unique_name('x' * 100) for _ in range(50)}
        self.assertEqual(len(names), 50)
        self.assertTrue(all(len(n) <= 50 for n in names))

    def test_malformed_env_is_user_error(self):
        p = argparse.ArgumentParser()
        cli._add_job_opts(p)
        with self.assertRaisesRegex(ValueError, 'KEY=VALUE'):
            cli._spec(p.parse_args(['--env', 'broken', 'train.py']))


class LifecycleTests(unittest.TestCase):
    def test_false_success_condition_does_not_mark_running_job_finished(self):
        self.assertEqual(core.job_state({'status': {'conditions': [
            {'type': 'Running', 'status': 'True'}, {'type': 'Succeeded', 'status': 'False'}]}}), 'Running')

    def test_replacement_job_is_not_followed(self):
        j = handle()
        j.uid = 'old'
        j.custom.get_namespaced_custom_object.return_value = {'metadata': {'uid': 'new'}}
        self.assertEqual(j.state(), 'Deleted')
        self.assertEqual(j.uid, 'old')

    def test_replacement_pod_is_not_read(self):
        j = handle()
        j.uid = 'old'
        j.core.read_namespaced_pod.return_value = pod(uid='new')
        with self.assertRaises(core.ApiException):
            list(j.stream_logs())
        j.core.read_namespaced_pod_log.assert_not_called()

    def test_delete_has_uid_precondition(self):
        with mock.patch.object(core.client, 'CustomObjectsApi') as api:
            self.assertTrue(core.delete_job('mtest', 'job', uid='old'))
        body = api.return_value.delete_namespaced_custom_object.call_args.kwargs['body']
        self.assertEqual(body.preconditions.uid, 'old')

    def test_replacement_delete_conflict_is_safe(self):
        with mock.patch.object(core.client, 'CustomObjectsApi') as api:
            api.return_value.delete_namespaced_custom_object.side_effect = core.ApiException(status=409)
            self.assertFalse(core.delete_job('mtest', 'job', uid='old'))

    def test_deleted_while_streaming_ends_without_traceback(self):
        j = handle()
        j.uid = 'old'
        j.wait_for_pod = mock.Mock(return_value='Running')
        j.core.read_namespaced_pod.return_value = pod()
        j.stream_logs = mock.Mock(side_effect=core.ApiException(status=404))
        j.state = mock.Mock(return_value='Deleted')
        j.exit_code = mock.Mock(return_value=None)
        self.assertEqual(j.follow(out=io.StringIO(), report=lambda _: None), 1)

    def test_completed_before_follow_still_prints_output(self):
        j = handle()
        j.uid = 'old'
        j.wait_for_pod = mock.Mock(return_value='Succeeded')
        j.core.read_namespaced_pod.return_value = pod(phase='Succeeded', code=0)
        j.stream_logs = mock.Mock(return_value=iter(['result=42', '[kjob now] exit code 0']))
        j.state = mock.Mock(return_value='Succeeded')
        output = io.StringIO()
        self.assertEqual(j.follow(out=output, report=lambda _: None), 0)
        self.assertIn('result=42', output.getvalue())

    def test_monitor_api_failure_does_not_cancel_training(self):
        j = handle()
        j.submit = mock.Mock()
        j.follow = mock.Mock(side_effect=core.ApiException(status=500))
        j.delete = mock.Mock()
        with self.assertRaises(core.ApiException):
            j.run(report=lambda _: None)
        j.delete.assert_not_called()

    def test_stream_response_is_closed(self):
        j = handle()
        j.uid = 'old'
        j.core.read_namespaced_pod.return_value = pod()
        response = j.core.read_namespaced_pod_log.return_value
        response.stream.return_value = [b'hello\n']
        self.assertEqual(list(j.stream_logs()), ['hello'])
        response.close.assert_called_once()

    def test_oom_reason_is_reported(self):
        j = handle()
        j.uid = 'old'
        p = pod(phase='Failed', code=137)
        p.status.container_statuses[0].state.terminated.reason = 'OOMKilled'
        j.core.read_namespaced_pod.return_value = p
        j.wait_for_pod = mock.Mock(return_value='Failed')
        j.stream_logs = mock.Mock(return_value=iter([]))
        j.state = mock.Mock(return_value='Failed')
        msgs = []
        self.assertEqual(j.follow(out=io.StringIO(), report=msgs.append), 137)
        self.assertIn('OOMKilled', msgs[-1])

    def test_api_connection_loss_is_a_clean_error(self):
        from urllib3.exceptions import MaxRetryError
        with mock.patch.object(core, '_load_config'), \
                mock.patch.object(cli, 'cmd_queue', side_effect=MaxRetryError(None, '/api', 'refused')), \
                mock.patch('sys.stderr', new_callable=io.StringIO) as err, self.assertRaises(SystemExit) as ex:
            cli.main(['queue'])
        self.assertEqual(ex.exception.code, 1)
        self.assertIn('keep running', err.getvalue())

    def test_ctrl_c_logs_does_not_dump_entire_saved_log(self):
        j = handle()
        j.state = mock.Mock(return_value='Running')
        j.stream_logs = mock.Mock(side_effect=KeyboardInterrupt)
        with mock.patch.object(cli, '_job_for', return_value=j):
            self.assertEqual(cli.cmd_logs(argparse.Namespace(name='job', follow=True, log_dir='/tmp')), 130)


if __name__ == '__main__':
    unittest.main()
