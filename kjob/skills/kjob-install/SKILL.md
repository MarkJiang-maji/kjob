---
name: kjob-install
description: Install, upgrade or troubleshoot the kjob CLI (ksub, ksrun, kqueue, ksinfo ...) inside a Kubeflow Notebook terminal on the CGU GPU clusters. Use when the user asks to install/set up/update kjob, when a kjob command is "not found", or before submitting GPU jobs on a notebook that does not have kjob yet.
---

# Installing kjob in a Kubeflow Notebook

kjob submits a script from a Kubeflow Notebook as a Kubernetes PyTorchJob that uses the same image, working
directory and shared volumes as the notebook. Commands are Slurm-like (ksub = sbatch, ksrun = srun, kqueue = squeue,
ksinfo = sinfo). Reply to the user in their language (most users write Traditional Chinese).

## 1. Check where you are

Run these first and report anything unexpected to the user instead of guessing:

```bash
cat /var/run/secrets/kubernetes.io/serviceaccount/namespace   # must exist: we are inside a Notebook pod
python3 --version                                            # 3.8 or newer
python3 -m pip --version
```

If the namespace file does not exist, you are not in a Kubeflow Notebook (e.g. a personal laptop or the cluster host).
kjob jobs are created *from* a Notebook; tell the user to open a terminal in their Notebook (JupyterLab ->
File -> New -> Terminal, or the terminal of the remote desktop image) and continue there.

## 2. Install

```bash
python3 -m pip install --user --upgrade https://github.com/MarkJiang-maji/kjob/archive/refs/heads/cgu-public.tar.gz
export PATH="$HOME/.local/bin:$PATH"
grep -q 'HOME/.local/bin' ~/.bashrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

* The archive URL needs no `git`. With git available this is equivalent:
  `python3 -m pip install --user --upgrade 'git+https://github.com/MarkJiang-maji/kjob.git@cgu-public'`.
* Inside an activated virtualenv/conda env: drop `--user` and use that env's `python -m pip`.
* `~/.local` is on the notebook workspace volume, so the install survives notebook restarts.
* It installs only kjob and the `kubernetes` Python client. It does not install or change PyTorch/CUDA.

## 3. Verify

```bash
ksub --help | head -5     # command exists
kinfo                     # notebook, namespace, inherited volumes, GPU quota
ksinfo                    # cluster GPU types and free GPUs (needs the administrator's status service)
```

* `kinfo` fails with a 403 error: the Notebook's service account lacks permissions. Only the cluster administrator can
  fix this; tell the user.
* `ksinfo` says "cluster-wide view is not available": the administrator has not deployed the status service.
  Job submission still works; only the cluster overview is missing.

## 4. Install the job-submission skill for the AI assistant (optional)

```bash
kjob install-skills                      # copies kjob-install and kjob-jobs into ~/.claude/skills
kjob install-skills --dir ~/.codex/skills   # another agent's skills directory
```

After that, use the `kjob-jobs` skill to submit and monitor jobs.

## Upgrade / uninstall

```bash
python3 -m pip install --user --upgrade --force-reinstall --no-deps https://github.com/MarkJiang-maji/kjob/archive/refs/heads/cgu-public.tar.gz
python3 -m pip uninstall kjob
```

## Troubleshooting

| Symptom | What to do |
|---|---|
| `ksub: command not found` after install | `export PATH="$HOME/.local/bin:$PATH"`, or run `python3 -m kjob ...` |
| `Can not perform a '--user' install` | you are in a virtualenv: install without `--user` |
| pip cannot reach GitHub / PyPI | ask the administrator for the wheel file, then `python3 -m pip install --user kjob-*.whl` |
| `externally-managed-environment` | the notebook image's Python is system-managed; use a virtualenv on `/home/jovyan` |
