# kjob 新手上手

這份教學帶你從安裝到提交第一個 GPU 工作，大約 10 分鐘。所有指令都在 **Notebook 的 Terminal** 執行
（JupyterLab：File → New → Terminal；遠端桌面 image：開啟終端機）。Notebook 本身不需要 GPU。

完整選項見 [docs/USER_GUIDE.md](docs/USER_GUIDE.md)。

## 1. 安裝

```bash
python3 -m pip install --user --upgrade https://github.com/MarkJiang-maji/kjob/archive/refs/heads/cgu-public.tar.gz
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

更新到最新版：重新執行第一行。使用虛擬環境時拿掉 `--user`。

## 2. 看看自己的環境和叢集

```bash
kinfo
```

```text
notebook  mynb-0  (namespace m1461023, node hs5800-2)
image     cguaicadmin/newlab-newpytorch:V1.0.8
workdir   /home/jovyan  (jobs start here unless -D is given)
volumes inherited by jobs:
  /home/jovyan             nfs 10.100.1.35:/Public/m1461023-mynb-workspace-pvc-...
  /mnt/groups/mylab        nfs ...
quota (used / hard):
  requests.nvidia.com/gpu      0 / 8
```

`volumes inherited by jobs` 列出的路徑，工作裡都看得到；程式和資料要放在這些地方。`quota` 是你的 namespace
能同時使用的 GPU 數（Notebook 自己占的 GPU 也算在內）。

```bash
ksinfo
```

```text
GPU_TYPE              NODES  GPUS  USED  FREE  WAITING_GPUS  GPU_MEM  NOTE
NVIDIA-RTX-A5000      1      10    10    0     -             24GB
Tesla-V100-SXM2-16GB  4      32    2     30    8 (1 jobs)    16GB
rtxpro6000            1      8     4     4     -             -        restricted (your namespace: no access)
```

選 `FREE` 有空位、記憶體夠的型號。`restricted` 表示需要另外開通，`no access` 的型號提交了也只會一直排隊。
想看某一種 GPU 是誰在用：`ksinfo -g rtxpro6000`；看全叢集排隊：`kqueue -A`。

> 若顯示 `cluster-wide view is not available`，代表管理員尚未啟用叢集狀態服務；提交工作不受影響。

## 3. 準備一個 GPU 測試程式

```bash
mkdir -p ~/kjob-demo && cat > ~/kjob-demo/gpu_test.py <<'PY'
import os, time, torch
print("node:", os.environ.get("KJOB_NODE"), "| GPU:", torch.cuda.get_device_name(0), flush=True)
x = torch.randn(4096, 4096, device="cuda")
for step in range(5):
    y = x @ x
    torch.cuda.synchronize()
    print(f"step {step + 1}/5 ok", flush=True)
    time.sleep(1)
PY
```

## 4. 提交並即時看輸出：`ksrun`

```bash
ksrun -G 1 -g v100 -c 2 --mem 8Gi ~/kjob-demo/gpu_test.py
```

```text
kjob: submitted gpu-test-1a2b3c4d5e6f (log: /home/jovyan/kjob-logs/gpu-test-1a2b3c4d5e6f.log)
kjob: gpu-test-1a2b3c4d5e6f running on hs5800-1
[kjob 2026-10-15T03:12:05Z] start gpu-test-1a2b3c4d5e6f on node hs5800-1
node: hs5800-1 | GPU: Tesla V100-SXM2-16GB
step 1/5 ok
...
kjob: gpu-test-1a2b3c4d5e6f Succeeded (exit code 0)
```

`-g v100` 換成 `ksinfo` 裡你可用的型號（例如 `a5000`、`a30`、`rtxpro6000`），或整個拿掉讓排程器挑任何空的 GPU。
`ksrun` 結束後會自動刪除工作；按 Ctrl-C 會取消工作。

## 5. 背景提交：`ksub`

長時間的訓練用 `ksub`，提交後就可以關掉瀏覽器：

```bash
ksub -J demo1 -G 1 -g v100 -c 2 --mem 8Gi -t 30 ~/kjob-demo/gpu_test.py
```

```text
Submitted job demo1  (log: /home/jovyan/kjob-logs/demo1.log)
```

`-t 30` 是時間上限 30 分鐘，從開始執行算起，時間到工作會被停止（格式也可用 `2:00:00`、`1-12:00:00`、`12h`）。
停止前程式會先收到 SIGTERM，有 30 秒可以存 checkpoint。

## 6. 查看工作

```bash
kqueue              # 自己的工作：狀態、節點、GPU 數
klogs demo1 -f      # 即時看輸出（還在排隊會先等），Ctrl-C 只停止觀看，工作繼續跑
kshow demo1         # 詳細資訊；排隊太久時看這裡的 event 找原因
kexec demo1 nvidia-smi   # 在執行中的工作裡跑指令，例如看 GPU 使用率；只打 kexec demo1 會開 shell 進去
kwait demo1         # 等到結束，回傳程式的結束代碼
kacct               # 歷史紀錄：是否成功、結束代碼、跑了多久
```

```text
NAME   SUBMITTED(UTC)    GPU  GPU_TYPE/NODE  STATE      EXIT  ELAPSED
demo1  2026-10-15 03:12  1    v100           COMPLETED  0     0m07s
```

輸出永久保存在 `~/kjob-logs/demo1.log`；程式寫到 `/home/jovyan` 的檔案就在你的 Notebook 裡。

## 7. 換成自己的程式

建議寫成一個 `.sh`，把資源寫在開頭（`#SBATCH` 是國網語法，也可用 `#KJOB` 寫 kjob 選項）：

```bash
#!/bin/bash
#SBATCH -J resnet50
#SBATCH --gres=gpu:v100:2
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#KJOB --shm 16Gi
set -e
cd /home/jovyan/project
source /home/jovyan/venv/bin/activate          # 若使用自己的虛擬環境
torchrun --standalone --nproc_per_node=2 train.py --epochs 100
```

```bash
ksub train.sh                 # 依照檔案開頭的設定提交
ksub -G 1 train.sh            # 命令列的選項優先，這次只要 1 張
```

要點：

* 路徑用絕對路徑，程式、資料、輸出放在 `kinfo` 列出的磁碟上。
* 多張 GPU 要程式本身支援（DDP / `torchrun`），否則只會用到第一張。
* PyTorch DataLoader 的 `num_workers` 大時加大 `--shm`（例如 `--shm 16Gi`）。
* 只在 Notebook 裝的套件：用 `pip install --user`（裝在 `~/.local`），或在 `/home/jovyan` 建虛擬環境。

## 8. 取消與清理

```bash
kcancel demo1      # 取消／刪除一個工作（日誌檔保留）
kclean             # 刪除這個 Notebook 已結束的工作物件
```

## 9. 常見問題

| 狀況 | 處理 |
|---|---|
| 一直 `Pending` | `kshow 名稱` 看 event。`Insufficient nvidia.com/gpu`：該型號滿了，等候或換型號；`exceeded quota`：namespace GPU 配額用完（關掉閒置的 GPU Notebook）；`untolerated taint`：沒有該節點權限 |
| `exit code 137, OOMKilled` | 記憶體不足，加大 `--mem` |
| `DeadlineExceeded` | 超過 `-t` 時間上限，加大 `-t` 或從 checkpoint 續跑 |
| `Bus error`、DataLoader 錯誤 | 加大 `--shm` |
| `No such file or directory` | 檔案不在共用磁碟或用了相對路徑 |
| `ModuleNotFoundError` | 套件裝在工作看不到的地方，見第 7 節 |
| `ksub: command not found` | `export PATH="$HOME/.local/bin:$PATH"` |

## 讓 AI 幫你提交

第 1 步之後第一次執行 kjob 指令時，skill 就自動裝到 `~/.claude/skills` 與 `~/.agents/skills`（畫面會出現一行
`kjob: installed AI skills ...`），Claude Code、Codex、OpenCode 等工具都讀得到。

之後在 Notebook 裡使用 Claude Code 等 AI 工具時，可以直接說「用 kjob 幫我把 train.py 交到有空的 V100 上，
2 張 GPU、跑 12 小時」，AI 會依照 skill 的規則選型號、設定資源、提交並回報工作名稱。你仍應確認 AI 選的資源合理。
