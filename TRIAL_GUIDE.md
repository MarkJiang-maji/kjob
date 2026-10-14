# kjob 試用流程

這是使用者自行安裝的試用版。你可以在 Notebook Terminal 安裝、提交工作，不需要重建共用 image。老師合併 PR 前，先從 Mark 的 fork 安裝。

## 開始前

- 使用有 Python 3.8 以上版本與 pip 的 Notebook；從 GitHub 安裝另外需要 git，安裝 wheel 不需要 git。
- Notebook 的服務帳號需要能讀取自己的 Pod、建立／讀取／刪除 PyTorchJob，以及查看工作 Pod、日誌和事件。
- 腳本、資料、工作目錄、日誌放在 `/home/jovyan` 的共用工作磁碟。底層 NFS 的 RWO workspace 可以由工具共用，不必因為試用而換磁碟；一般 RWO block volume 不適用。
- GPU 訓練需要工作 image 中有 CUDA/PyTorch，或共用磁碟裡有相容的環境。工具安裝不會替你安裝 PyTorch。
- Notebook 本身建議不占 GPU；否則它可能用完 namespace 的 GPU 配額。

工具會自動使用你所在的 namespace。安裝成功不代表其他 namespace 的權限、儲存與配額已驗證；本次實測位置是 `mtest/kjob-test`。

## 1. 安裝

從 Mark 的 fork 指定 `cgu-a30-fixes` 試用分支安裝。也可安裝管理員提供的 wheel，完整步驟見 [試用與正式上線說明](DEPLOYMENT_GUIDE.md)。

在 Notebook 的 Terminal 執行：

```bash
python3 --version
python3 -m pip install --user --upgrade 'git+https://github.com/MarkJiang-maji/kjob.git@cgu-a30-fixes'
export PATH="$HOME/.local/bin:$PATH"
ksub --help
kinfo
```

`cgu-a30-fixes` 是試用分支，更新時內容可能改變。若要重現固定版本，可以把 `@cgu-a30-fixes` 換成提供給你的完整 commit SHA。試用版尚不代表老師已合併或全平台正式部署。

如果目前在虛擬環境內，請使用該環境的 `python -m pip install --upgrade ...`（省略 `--user`），並保持環境啟用。CLI 與訓練程式需要的 Python 路徑也要在工作 image 或共用磁碟上可用。不要用宿主機的私人虛擬環境路徑提交。

## 2. 建立兩個小範例

下列指令在 `/home/jovyan/kjob-demo` 建立檔案；同名檔案已存在就保留，不覆蓋：

```bash
python3 - <<'PY'
from pathlib import Path
folder = Path('/home/jovyan/kjob-demo')
folder.mkdir(exist_ok=True)
scripts = {
    'hello.py': '''import os, json
print(json.dumps({"result": 42, "node": os.environ.get("KJOB_NODE")}), flush=True)
''',
    'gpu.py': '''import os, json
from pathlib import Path
import torch
assert torch.cuda.is_available(), "目前工作環境無法使用 CUDA"
x = torch.ones((32, 32), device="cuda")
result = {"gpu": torch.cuda.get_device_name(0), "node": os.environ.get("KJOB_NODE"), "checksum": float((x @ x).sum())}
out = Path('/home/jovyan/kjob-demo/results')
out.mkdir(exist_ok=True)
(out / (os.environ['KJOB_NAME'] + '.json')).write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
'''
}
for name, text in scripts.items():
    target = folder / name
    if target.exists():
        print('保留既有檔案:', target)
    else:
        target.write_text(text)
        print('已建立:', target)
PY
```

## 3. 先試 CPU

```bash
ksub -G 0 -c 1 --mem 2Gi /home/jovyan/kjob-demo/hello.py
kqueue
```

提交會顯示完整工作名稱，例如 `hello-xxxxxxxxxxxx`。把下面的文字換成你的名稱：

```bash
JOB_ID='換成剛才回傳的完整工作名稱'
kwait "$JOB_ID" --keep
klogs "$JOB_ID"
```

看到 `result: 42` 代表範例成功。`kwait --keep` 完成後不立即刪工作，仍受提交時的 TTL 設定影響。日誌預設保存在 `~/kjob-logs/工作名稱.log`。

## 4. 再試 GPU

在目前 .6 平台，整張 A30：

```bash
ksub -G 1 -g a30 -c 1 --mem 2Gi /home/jovyan/kjob-demo/gpu.py
```

MPS 共享 A30 需管理員先確認節點健康與 UID 相容；目前不建議新手試用時選擇：

```bash
ksub -G 1 -g a30-shared -c 1 --mem 2Gi /home/jovyan/kjob-demo/gpu.py
```

共享模式目前只支援一個資源單位；這版只允許 UID 0 Notebook 提交共享工作，但 root 並不保證 MPS 可用。2026-10-14 檢查發現 `gpunode051` 的 MPS server 是 UID 1000，root 工作會卡在 CUDA 初始化。先使用整卡模式。需要兩張 GPU 時使用整卡模式，且 namespace 要有至少兩個 GPU 配額。

工作完成後，結果在 `/home/jovyan/kjob-demo/results/工作名稱.json`。更新 `JOB_ID`，再用 `kwait "$JOB_ID" --keep` 或 `klogs "$JOB_ID"` 查看。

若要指定節點，也一併寫明型號：

```bash
ksub -G 1 -g a30-shared -w gpunode051 -c 1 --mem 2Gi /home/jovyan/kjob-demo/gpu.py
```

`gpunode051` 目前有上述 MPS 問題，這是語法範例，修復並驗證前請不要實際提交。其他叢集請使用管理員提供的 node 與型號。

## 5. 換成自己的程式

```bash
ksub -G 1 -g a30 -c 2 --mem 4Gi /home/jovyan/start.sh
```

`start.sh` 需要自行準備，內容例如：

```bash
#!/bin/bash
set -e
cd /home/jovyan/project
# 如果已在共用磁碟建立相容環境，可在這裡啟用。
# source /home/jovyan/myenv/bin/activate
python3 -u train.py
```

`kjob` 不讀取 `#SBATCH` 註解，CPU/GPU/記憶體要放在 `ksub` 指令上。這個版本沿用 Notebook image；舊 notebook-sbatch 的 `--image`、`sattach` 沒有對應功能。

## 查進度與取消

| 目的 | 指令 |
|---|---|
| 看工作清單 | `kqueue` |
| 看輸出 | `klogs "$JOB_ID"` |
| 持續看輸出 | `klogs "$JOB_ID" -f` |
| 等待完成並保留工作 | `kwait "$JOB_ID" --keep` |
| 取消指定工作 | `kcancel "$JOB_ID"` |
| 提交並即時看輸出 | `ksrun -G 1 -g a30 -c 1 --mem 2Gi /home/jovyan/kjob-demo/gpu.py` |

`klogs` 或 `kwait` 按 Ctrl-C 會停止查看，工作繼續；`ksrun` 按 Ctrl-C 會取消工作。`kclean` 預設只清目前 Notebook 的已完成工作；日誌和程式檔案不會因此刪除。

申請超過 namespace 總配額會直接提示錯誤；暫時配額滿會等待既有 operator 重試。這不是嚴格先進先出的排隊系統。

## 本次不調整的平台設定

既有 Notebook 的 `/dev/shm` 啟動問題、按 GPU 型號分配額、全平台 Pod/NFS 權限 webhook 與共用 image 預裝都保留待後續處理。工具的檢查不能取代服務端授權；這次先供受控環境的使用者試用，正式開放前仍需評估平台政策。
