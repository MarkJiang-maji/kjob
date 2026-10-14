# kjob

Run a script as a Kubeflow **PyTorchJob** that looks like the notebook you are in: same image, same working
directory, same shared volumes. Output is streamed live, saved to `~/kjob-logs/<name>.log`, and the job is deleted
when it finishes. Slurm-like commands sit on top.

## Install

```bash
python3 -m pip install --user git+https://github.com/wycc/kjob.git   # needs the `kubernetes` package (>=28)
export PATH=~/.local/bin:$PATH                             # kjob, ksub, ksrun, kqueue, ...
```

pip installs `kjob` and the short commands (`ksub`, `ksrun`, `kwait`, `kqueue`, `klogs`, `kcancel`, `kclean`,
`kinfo`). From a source checkout, `pip install -e /path/to/kjob` does the same; the `bin/` wrappers also work and run
`python3` (set `KJOB_PYTHON` to use another interpreter, e.g. a virtualenv).

For a pre-merge trial of the CGU fixes, see [the trial guide](TRIAL_GUIDE.md). Python 3.8+ is required.
For wheel distribution, namespace checks and a pinned formal release, see [the deployment guide](DEPLOYMENT_GUIDE.md).

## Commands

| Slurm | kjob | What it does |
|---|---|---|
| `srun` | `ksrun [opts] script.py args...` | submit, stream output live, delete the job, exit with the script's exit code (Ctrl-C cancels) |
| `sbatch` | `ksub [opts] script.py args...` | submit and return; the job deletes itself `--ttl` seconds after it ends |
| — | `kwait NAME` | follow a submitted job until it ends, then delete it |
| `squeue` | `kqueue [-w] [--all]` | list jobs: state, pod phase, node, GPUs, CPUs, age, command |
| — | `klogs NAME [-f]` | print / follow output (falls back to the saved log file once the pod is gone) |
| `scancel` | `kcancel NAME... \| --finished \| --all-mine [--all-notebooks]` | delete jobs |
| — | `kclean [--all-notebooks]` | delete finished jobs |
| `sinfo` | `kinfo` | what jobs inherit from the notebook, quota usage, GPU types |

Job options (`ksub` / `ksrun`):

| Option | Default | Meaning |
|---|---|---|
| `-J, --job-name` | script name + unique suffix | job name |
| `-G, --gpus` | 1 | GPUs (0 = CPU-only; no GPU request is made) |
| `-c, --cpus` / `--mem` / `--shm` | 1 / 2Gi / 256Mi | CPU cores, memory, `/dev/shm` size |
| `-g, --gpu-type` | any whole GPU | `a5000`, `v100`, `a30`, `a30-shared` (comma list) or a raw `nvidia.com/gpu.product` value |
| `-w, --nodelist` | — | pin to a host; specify `-g` too if the Notebook cannot read node labels |
| `-x, --exclude` | — | hosts to avoid (comma list) |
| `--python` | the calling Python | interpreter for `.py` scripts, e.g. `.venv-v100/bin/python` for V100 |
| `--env KEY=VALUE` | — | extra environment variables (repeatable) |
| `-D, --chdir` | current directory | working directory inside the job |
| `-o, --log-dir` | `~/kjob-logs` | where output is saved |
| `--ttl` | 3600 | seconds a finished, unmonitored job is kept before deletion |

`.py` scripts run with the chosen Python, `.sh` with bash, anything else is executed directly. Put `--` before
script arguments that look like kjob options.

### Examples

```bash
# fine-tune on a V100 with the V100-compatible environment, watching the output
ksrun -J ft-papila -g v100 -c 2 --mem 24Gi --python /home/jovyan/dinov2/.venv-v100/bin/python \
      /home/jovyan/dinov2/finetune.py --weights meta --name test --datasets PAPILA --out /tmp/x

# pretraining on the RTX node, in the background
ksub -J pretrain-x -w rtxpro6000 -G 4 -c 40 --mem 200Gi --shm 96Gi pretrain.py --epochs 30
kqueue -w
kwait pretrain-x
```

## Python API

```python
from kjob import JobSpec, KJob

job = KJob(JobSpec(name="probe-1", script="linear_probe.py", args=["--weights", "meta", "--name", "x"],
                   gpus=1, cpus=2, memory="24Gi", gpu_types=["v100"],
                   interpreter="/home/jovyan/dinov2/.venv-v100/bin/python"))
code = job.run()          # submit + stream + delete; returns the script's exit code
# or: job.submit(); ...; job.follow(); job.delete()
```

## How the notebook environment is copied

* Image, working directory and Pod/container user and group settings come from the notebook pod (read with the Kubernetes API).
* Every volume mounted into the notebook container is copied, except:
  * service-account tokens and Istio sidecar volumes,
  * volumes injected by a matching namespace **PodDefault** (e.g. `/nfs`): selector labels are carried to the job, the PodDefault adds them again, and
    the admission webhook rejects duplicates.
* ReadWriteOnce claims backed by NFS are mounted as the NFS export found in `/proc/mounts`, so a job can run on a
  different node than the notebook. Read-only settings are retained. NFS-backed PVCs with `subPath` or
  `subPathExpr` are rejected before submission; mount the full volume. This does not make arbitrary RWO block storage shareable.
* `emptyDir` volumes (e.g. `/dev/shm`) are recreated empty; the notebook's `/tmp` is **not** visible to jobs.
* GroupShare / NamespaceShare volumes (`gs-*`, `ns-*`) are never copied: the PodDefault adds the current ones. A
  notebook started before its groups changed may still mount old ones (`kinfo` lists them as outdated); jobs do not
  get them.
* `kinfo` shows exactly what is copied.

## Notes

* `kqueue` lists every kjob job in the namespace with the notebook it came from. `kcancel --finished/--all-mine` and
  `kclean` only delete jobs submitted from the current notebook unless `--all-notebooks` is given; `kcancel NAME`
  deletes that job.
* CPU-only jobs (`-G 0`) get `NVIDIA_VISIBLE_DEVICES=void` and `CUDA_VISIBLE_DEVICES=-1`, so CUDA images do not see the
  node's GPUs. Job pods do not mount a service-account token.
* GPU sharing: nodes whose product label ends in `-SHARED` (e.g. `NVIDIA-A30-SHARED`, `nvidia.com/gpu.sharing-strategy=mps`)
  hand out MPS slices, and the device plugin mounts its own `/dev/shm` there, so jobs on them do not get the `--shm`
  volume. Whole-GPU requests avoid MPS nodes, including pinned requests. Use `-g a30-shared` for a slice. Both
  count as one `nvidia.com/gpu` in the namespace quota.

* Exit codes: the wrapper prints `[kjob ...] exit code N` as the last log line; it is used when the pod status
  is no longer available.
* Namespace quota counts pending pods too; a job waiting for a busy GPU type still holds quota.
* GPU product labels known so far are listed in `kjob/core.py` (`GPU_PRODUCTS`); any label value also works.

## Submission checks and compatibility

* Defaults are 1 GPU resource unit, 1 CPU, 2 GiB RAM and a 256 MiB shared-memory limit. CPU-only work must explicitly use `-G 0`.
* Requests larger than a readable namespace hard quota fail before creating a job. Temporarily full quotas still allow operator retry; this is not FIFO or a fair queue.
* This installation only enables shared MPS jobs for UID 0 Notebooks; the administrator must also verify the server UID and health. Root alone does not guarantee compatibility: on 2026-10-14 the MPS server on `gpunode051` ran as UID 1000 and root CUDA clients stalled. Whole A30 GPUs are recommended for the trial. kjob never changes a non-root caller to root. MPS requests larger than one resource unit and mixed whole/shared type lists are rejected.
* A pinned GPU node without `-g` is inspected when RBAC permits. If it cannot be inspected, specify `-g a30` or `-g a30-shared` explicitly; no extra cluster-wide permissions are required.
* Scripts, the working directory and logs must be on shared mounts; logs must be writable. Runtime-owned environment variables cannot be overridden. These client checks are not a replacement for server-side GPU or NFS authorization.
* Monitoring and cleanup bind to the job UID, including deletion preconditions. A replaced job is treated as deleted. Unexpected monitoring/API failures leave training running; an explicit Ctrl-C during `ksrun` cancels it.
* Script failures use replica `restartPolicy: Never` without an explicit `backoffLimit`. Older operators can delete a failed Pod before recording the failure when `backoffLimit: 0` is set, then recreate the Pod during reconciliation. Omitting that limit keeps the normal failure path and preserves the failed Pod for inspection.
* This utility uses the Notebook image and an interpreter available at the same path in the job. It does not copy packages installed only in the live Notebook container, parse `#SBATCH` directives, provide `sattach`, or offer an image override. Pass resources on the `ksub` command line.
* Existing Notebook `/dev/shm` failures, per-GPU-model quotas and platform webhook coverage are separate platform work, not installed by this package.
