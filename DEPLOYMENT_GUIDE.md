# kjob：試用與正式上線

這個功能讓使用者在 Notebook 裡寫好程式後，提交一個新的 PyTorchJob 到叢集執行。Notebook 不必一直占著 GPU；工作可以跑在其他節點，結果寫回同一個工作磁碟。

**對使用者而言，安裝 kjob 後就能用 `ksub`、`kqueue`、`klogs` 等指令。** 前提是該 namespace 已有 PyTorchJob 權限、可共用的工作磁碟和足夠配額。不是每個 Notebook 安裝成功就代表 GPU 程式一定能跑。

## 目前在哪個階段

2026-10-14 檢查的程式位於 `/home/mark/work/kjob`，試用分支為 `MarkJiang-maji/kjob:cgu-a30-fixes`，套件版本為 `0.3.0`。試用請指定此分支；老師審核合併前，直接從老師的 repo 或 fork 的 `main` 安裝仍會拿到原始版本。

建議先在少數 namespace 試用 CPU 與整張 A30。MPS 共享 GPU 暫不列入推薦流程：`gpunode051` 的 MPS server UID 與 root 工作不相容，CUDA 初始化曾卡住；這次沒有重啟或修改共用 MPS 服務。

檢查時也發現舊 operator 配合 `backoffLimit: 0` 會刪除失敗 Pod，並可能在狀態更新前重建它，使腳本執行兩次。已改為保留 `restartPolicy: Never`、省略 `backoffLimit`，走正常的失敗處理流程；不需要升級共用 operator。

完整新手範例見 [TRIAL_GUIDE.md](TRIAL_GUIDE.md)。本文件主要說明怎麼把套件提供給使用者，以及正式上線要做什麼。
本次檢查範圍、結果與尚未完成的部分見 [REVIEW_STATUS.md](REVIEW_STATUS.md)。

## 一、用 wheel 試用

wheel 是已打包好的 Python 安裝檔，不必等 GitHub push，也不用重建 Notebook image。

### 管理員：把檔案放進 Notebook

在 **mark 主機的 Terminal** 執行，先啟動 `mtest/kjob-test` Notebook：

```bash
KJOB_NAMESPACE=mtest
KJOB_NOTEBOOK_POD=kjob-test-0
KJOB_NOTEBOOK_CONTAINER=kjob-test

sudo -H kubectl --as="system:serviceaccount:${KJOB_NAMESPACE}:default-editor" \
  -n "$KJOB_NAMESPACE" cp \
  /home/mark/work/kjob/dist/kjob-0.3.0-py3-none-any.whl \
  "$KJOB_NOTEBOOK_POD:/home/jovyan/kjob-0.3.0-py3-none-any.whl" \
  -c "$KJOB_NOTEBOOK_CONTAINER"
```

這裡的 sudo 只用於既有 kubectl 管理權限；不需要 `sudo -i`，也不會安裝宿主機套件或修改全平台設定。若你的 kubectl 本來就有該 namespace 的權限，可以省略 sudo 和 impersonation。

其他 namespace 換成實際的 namespace、Pod 與主容器名稱即可；請只操作已獲授權的 Notebook。也可以從 `kjob-test` 的檔案介面下載 wheel，再上傳到其他 Notebook，不必給使用者 kubectl 權限。

### 使用者：在 Notebook Terminal 安裝

```bash
python3 --version
python3 -m pip install --user --upgrade /home/jovyan/kjob-0.3.0-py3-none-any.whl
export PATH="$HOME/.local/bin:$PATH"
python3 -m pip show kjob
ksub --help
kinfo
```

需要 Python 3.8 以上。這條指令只安裝目前 Notebook 使用者的套件；wheel 的相依套件仍需能從 Python package index 下載，或已經安裝。若正在使用 virtualenv，請改用該環境的 `python -m pip install --upgrade ...`，省略 `--user`。

如果已安裝同版本 `0.3.0`，但管理員提供了更新的 wheel，請只重裝 kjob，避免改動既有相依套件：

```bash
python3 -m pip install --user --force-reinstall --no-deps /home/jovyan/kjob-0.3.0-py3-none-any.whl
```

第一次安裝請用前面的正常安裝指令，讓 pip 安裝相依套件。

### 提交一個最小測試

在 Notebook Terminal 執行，建立一個不會覆蓋既有檔案的測試腳本：

```bash
cd /home/jovyan
DEMO_SCRIPT="$(mktemp /home/jovyan/kjob-hello-XXXXXX.py)"
cat > "$DEMO_SCRIPT" <<'PY'
import os
print('hello from', os.environ.get('KJOB_NODE'), flush=True)
PY
ksrun -G 0 -c 1 --mem 2Gi "$DEMO_SCRIPT"
```

看到 `hello from ...` 和 `Succeeded (exit code 0)` 就表示 CPU 工作完成。`ksrun` 結束後會刪除這個工作，但保留腳本與工作磁碟上的日誌。

換成自己的 GPU 訓練腳本：

```bash
cd /home/jovyan
ksub -G 1 -g a30 -c 1 --mem 2Gi /home/jovyan/start.sh
kqueue
```

`start.sh` 是使用者自己的程式。提交後會顯示工作名稱；把下面的 `JOB_ID` 換成該名稱：

```bash
JOB_ID='填入剛才回傳的工作名稱'
kwait "$JOB_ID" --keep
klogs "$JOB_ID"
# 需要取消時：
kcancel "$JOB_ID"
```

`--keep` 保留完成的工作供查看，但仍受提交時的 TTL 影響。Notebook 建議使用 CPU，把 namespace 的 GPU 配額留給工作。Notebook 與工作會共用 CPU、RAM、GPU 配額；mtest 目前上限是一個 GPU 資源單位。

## 二、使用者直接從 GitHub 安裝

### 管理員：發布試用分支並建立 PR

試用分支可透過已授權的 GitHub 連接器發布。若之後改用 Terminal 推送，先在該工作機完成 **MarkJiang-maji GitHub 登入**，並從 GitHub clone 發布分支，再修改及推送：

```bash
git clone --branch cgu-a30-fixes https://github.com/MarkJiang-maji/kjob.git kjob-trial
cd kjob-trial
git status --short
# 修改、檢查並 commit 後：
git push origin cgu-a30-fixes
```

這段 clone 指令適用於還沒有 `kjob-trial` 目錄的工作機。不要把 token 寫在網址或指令裡。本機原始開發 commits 與 GitHub 發布 commit 的歷史可能不同；後續修改建議從發布分支 clone，避免直接推送造成 non-fast-forward。

push 完成後，到以下頁面建立 PR，base 為 `wycc/kjob:main`，head 為 `MarkJiang-maji/kjob:cgu-a30-fixes`：

[建立給老師審核的 PR](https://github.com/wycc/kjob/compare/main...MarkJiang-maji:kjob:cgu-a30-fixes?expand=1)

這一步讓修改可下載與審核，不會重啟平台或替所有 Notebook 安裝套件。

### 使用者：安裝試用分支

在任何已符合條件的 Notebook Terminal 執行：

```bash
python3 -m pip install --user --upgrade 'git+https://github.com/MarkJiang-maji/kjob.git@cgu-a30-fixes'
export PATH="$HOME/.local/bin:$PATH"
kinfo
```

從 Git 安裝需要 Notebook 裡有 git。試用更新時若套件版本仍相同，請用管理員提供的固定 commit，必要時再用 `--force-reinstall --no-deps` 重裝 kjob。

工具會自動讀取所在 Notebook 的 namespace，不必另寫 `--namespace mtest`。到其他 namespace 的 Notebook 安裝，就會在那個 namespace 提交；不會跨到 mtest 借配額。節點可以跨，namespace 配額仍然不變。

## 三、正式上線：固定老師核准的版本

推薦先採用 **老師合併 PR → 固定 commit → 使用者自行 pip install**。這就是這個工具的一種正式發布方式，不必先重建 image。

### 管理員要完成的事情

1. PR 合併後，在核准的版本重跑測試，並於目標 namespace 驗證 CPU、整卡 GPU、配額等候、取消和日誌。
2. 記下核准的完整 commit SHA，發布統一安裝指令；不要讓正式使用者追著可變動的試用分支。
3. 確認目標 namespace 的服務帳號、儲存與配額。若要限制群組目錄或 GPU 型號，限制必須放在平台服務端；kjob 客戶端檢查不能取代授權。
4. MPS 若要列入正式支援，另安排節點健康、server UID 與不同 image 的驗證。目前建議正式流程先以 CPU、整卡 GPU 為範圍。

### 使用者正式安裝指令

以下要等 PR 合併，並把佔位文字換成 **老師核准的 40 字元 commit SHA**：

```bash
KJOB_APPROVED_COMMIT='填入老師核准的完整commitSHA'
python3 -m pip install --user --upgrade \
  "git+https://github.com/wycc/kjob.git@${KJOB_APPROVED_COMMIT}"
export PATH="$HOME/.local/bin:$PATH"
python3 -m pip show kjob
kinfo
```

若先前已裝同版本的試用版，接著執行以下指令，確保換成核准的程式內容；相依套件由前一條正常安裝指令處理：

```bash
python3 -m pip install --user --force-reinstall --no-deps \
  "git+https://github.com/wycc/kjob.git@${KJOB_APPROVED_COMMIT}"
```

之後提交工作的方法和試用相同。`pip install kjob` 這種只寫名稱的指令，是從 Python package index 安裝同名套件；**目前沒有確認這個專案已發布到 PyPI**，所以使用提供的 Git URL 或 wheel。

### 若希望新 Notebook 不必手動安裝

可以把核准的 wheel 預裝到新的 Notebook image，再把新 image 加進平台選單。這是第二階段選項，會涉及 image 建置與平台設定；目前沒有執行。

先準備已驗證的 wheel，Dockerfile 範例如下。`BASE_IMAGE` 使用你原來的 Notebook image；`NOTEBOOK_USER` 必須保持原 image 的執行使用者，不能為了 kjob 改成 root：

```dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG NOTEBOOK_USER
USER root
COPY kjob-0.3.0-py3-none-any.whl /tmp/kjob-0.3.0-py3-none-any.whl
RUN python3 -m pip install --no-cache-dir /tmp/kjob-0.3.0-py3-none-any.whl && \
    python3 -m pip install --no-cache-dir --force-reinstall --no-deps /tmp/kjob-0.3.0-py3-none-any.whl && \
    rm /tmp/kjob-0.3.0-py3-none-any.whl
USER ${NOTEBOOK_USER}
```

建置前需確認基底 image 的 `python3` 是 Jupyter 所用環境，套件安裝到該環境。此範例在 image 建置時安裝系統套件，完成後恢復原執行使用者；不改動宿主機的 Python。

在有 image 建置／registry 權限的工作機，填入實際值後執行：

```bash
KJOB_BASE_IMAGE='填入目前Notebook的image'
KJOB_IMAGE_USER='填入基底image原來的USER；原本未設定則填0'
KJOB_NEW_IMAGE='填入你的registry/新image名稱:固定版本'
cd /home/mark/work/kjob/dist
docker build -f ../deploy/Dockerfile --build-arg BASE_IMAGE="$KJOB_BASE_IMAGE" \
  --build-arg NOTEBOOK_USER="$KJOB_IMAGE_USER" -t "$KJOB_NEW_IMAGE" .
docker push "$KJOB_NEW_IMAGE"
```

這些指令只發布 image，不會讓平台自動改用它。之後需由管理員更新 Notebook image 選單，先建一個測試 Notebook，再開放選用。現有 Notebook 不會因為 push image 就自動更新。

## 四、換 namespace 前怎麼確認

一般使用者先在目標 Notebook 執行 `kinfo`，再跑前面的 CPU 小測試。管理員可以在主機進一步做唯讀檢查：

```bash
KJOB_NAMESPACE=mtest  # 改成要試用的 namespace
KJOB_SERVICE_ACCOUNT="system:serviceaccount:${KJOB_NAMESPACE}:default-editor"

sudo -H kubectl get crd pytorchjobs.kubeflow.org
sudo -H kubectl -n "$KJOB_NAMESPACE" get resourcequota
sudo -H kubectl -n "$KJOB_NAMESPACE" get pods,pvc
sudo -H kubectl --as="$KJOB_SERVICE_ACCOUNT" -n "$KJOB_NAMESPACE" auth can-i create pytorchjobs.kubeflow.org
sudo -H kubectl --as="$KJOB_SERVICE_ACCOUNT" -n "$KJOB_NAMESPACE" auth can-i get pods
sudo -H kubectl --as="$KJOB_SERVICE_ACCOUNT" -n "$KJOB_NAMESPACE" auth can-i get pods --subresource=log
sudo -H kubectl --as="$KJOB_SERVICE_ACCOUNT" -n "$KJOB_NAMESPACE" auth can-i list poddefaults.kubeflow.org
```

Notebook 若使用不同 ServiceAccount，請改成其實際名稱。完整操作還需要 list/get/delete PyTorchJob、list Pod、list events、list ResourceQuota；不需要額外開放讀取所有 nodes，也不需要 root shell。權限不足時應確認平台既有 RoleBinding，不要直接給 cluster-admin。

工作沿用 Notebook 的 image 與共享磁碟，**不會複製只安裝在 Notebook 容器可寫層裡的套件或臨時檔案**。將程式、資料與相容的 virtualenv 放在 `/home/jovyan` 共用工作磁碟，或使用本來已具備訓練環境的 image。這台平台的 NFS-backed RWO workspace 可共用；一般 RWO block volume 不能套用同樣做法。

`-w` 指定節點後，該節點本身也必須有足夠資源。namespace 還有 GPU 配額，不代表指定節點有空卡；測試時 `gpunode052` 曾因 `Insufficient nvidia.com/gpu` 等候。一般先省略 `-w`，讓排程器找可用的整卡節點。

## 五、在 .4、.25 或其他叢集使用

同一份 kjob 可以從 GitHub 安裝到不同叢集的 Notebook；安裝的是 Python 指令工具，不需要為 kjob 另外建立 Deployment。工具透過所在 Notebook 的 Kubernetes API 和 ServiceAccount 提交工作，所以在哪個叢集、哪個 namespace 的 Notebook 執行，就使用那裡的資源與配額。

**本次只在 .6 的 mtest 實測，尚未驗證 .4、.25。** 每個目的叢集先確認下列條件，再選一個 Notebook 做小型測試：

| 條件 | 要確認的內容 |
|---|---|
| Training operator | 有正常運作的 operator，並提供 `kubeflow.org/v1` 的 `PyTorchJob`；新版只提供其他 API 時需要另做相容調整 |
| Notebook 權限 | ServiceAccount 能讀取 Notebook Pod、操作 PyTorchJob、查看工作 Pod／日誌／事件與配額；具體唯讀檢查見上一節 |
| 工作磁碟 | 程式與資料能由工作 Pod 共用；目前支援流程包括底層 NFS 的完整 PVC 掛載及可共用的 RWX 磁碟，不能假設一般 RWO block volume 可跨節點使用 |
| 執行環境 | Python 3.8+、pip，從 Git 安裝另需 git；工作 image 或共用磁碟上的環境包含訓練所需套件 |
| GPU | 有 NVIDIA device plugin、可用 GPU 與 namespace 配額；型號標籤和共享設定按該叢集確認 |

在目的叢集的 **Notebook Terminal** 執行：

```bash
python3 -m pip install --user --upgrade 'git+https://github.com/MarkJiang-maji/kjob.git@cgu-a30-fixes'
export PATH="$HOME/.local/bin:$PATH"
kinfo
```

先跑本文件的 CPU 小測試。CPU 通過、管理員確認 GPU 設定後，可以用自己的腳本測試 GPU：

```bash
ksub -G 1 -c 1 --mem 2Gi /home/jovyan/start.sh
kqueue
```

這裡沒有指定型號或節點；目前預設會排除標記為 MPS 的節點。若要指定 GPU，可使用 `-g a5000`、`-g v100`、`-g a30`，或 `-g '實際的nvidia.com/gpu.product標籤值'`。只有目的叢集標籤相符才會排到；不要直接複製 .6 的 A30 型號或 `gpunode050` 等節點名稱。

若目的叢集使用 time-slicing 或不同的 MPS 設定，先請管理員檢查；目前「避開 MPS」不等於排除所有共享模式，不應直接把每個 GPU 資源單位解讀成一整張卡。`.4`、`.25` 若要正式開放，需各自驗證 CPU、GPU、等候、取消及日誌，再使用老師核准的固定 commit 安裝。

## 參考

- [pip 從 Git 分支、tag、commit 安裝](https://pip.pypa.io/en/stable/topics/vcs-support/)
- [Kubernetes namespace 配額](https://kubernetes.io/docs/concepts/policy/resource-quotas/)
- [GitHub fork 與 PR](https://docs.github.com/en/pull-requests/how-tos/work-with-forks)
