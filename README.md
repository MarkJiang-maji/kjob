# kjob

Run a script as a Kubeflow **PyTorchJob** that looks like the notebook you are in: same image, same working
directory, same shared volumes. Output is streamed live, saved to `~/kjob-logs/<name>.log`, and the job is deleted
when it finishes. Slurm-like commands sit on top.

## Install

```bash
uv pip install --python /home/jovyan/dinov2/.venv/bin/python -e /home/jovyan/kjob   # needs the `kubernetes` package
export PATH=/home/jovyan/kjob/bin:$PATH                                              # ksub, ksrun, kqueue, ...
```

The `bin/` wrappers use `/home/jovyan/dinov2/.venv/bin/python`; set `KJOB_PYTHON` to use another interpreter.

## Commands

| Slurm | kjob | What it does |
|---|---|---|
| `srun` | `ksrun [opts] script.py args...` | submit, stream output live, delete the job, exit with the script's exit code (Ctrl-C cancels) |
| `sbatch` | `ksub [opts] script.py args...` | submit and return; the job deletes itself `--ttl` seconds after it ends |
| — | `kwait NAME` | follow a submitted job until it ends, then delete it |
| `squeue` | `kqueue [-w] [--all]` | list jobs: state, pod phase, node, GPUs, CPUs, age, command |
| — | `klogs NAME [-f]` | print / follow output (falls back to the saved log file once the pod is gone) |
| `scancel` | `kcancel NAME... \| --finished \| --all-mine` | delete jobs |
| — | `kclean` | delete finished jobs |
| `sinfo` | `kinfo` | what jobs inherit from the notebook, quota usage, GPU types |

Job options (`ksub` / `ksrun`):

| Option | Default | Meaning |
|---|---|---|
| `-J, --job-name` | script name + time | job name |
| `-G, --gpus` | 1 | GPUs (0 = CPU-only; no GPU request is made) |
| `-c, --cpus` / `--mem` / `--shm` | 4 / 32Gi / 16Gi | CPU cores, memory, `/dev/shm` size |
| `-g, --gpu-type` | any | `a5000`, `v100` (comma list) or a raw `nvidia.com/gpu.product` value |
| `-w, --nodelist` | — | pin to a host, e.g. `rtxpro6000` |
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

* Image, working directory and pod security context come from the notebook pod (read with the Kubernetes API).
* Every volume mounted into the notebook container is copied, except:
  * service-account tokens and Istio sidecar volumes,
  * volumes injected by a namespace **PodDefault** (e.g. `/nfs`): the PodDefault adds them to the job again, and
    the admission webhook rejects duplicates.
* ReadWriteOnce claims backed by NFS are mounted as the NFS export found in `/proc/mounts`, so a job can run on a
  different node than the notebook.
* `emptyDir` volumes (e.g. `/dev/shm`) are recreated empty; the notebook's `/tmp` is **not** visible to jobs.
* `kinfo` shows exactly what is copied.

## Notes

* Exit codes: the wrapper prints `[kjob ...] exit code N` as the last log line; it is used when the pod status
  is no longer available.
* Namespace quota counts pending pods too; a job waiting for a busy GPU type still holds quota.
* GPU product labels known so far are listed in `kjob/core.py` (`GPU_PRODUCTS`); any label value also works.
