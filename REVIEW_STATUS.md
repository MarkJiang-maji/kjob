# 2026-10-14 檢查紀錄

檢查對象是 `/home/mark/work/kjob` 的 `cgu-a30-fixes` 分支，包含 Claude 的 `988bfbf`、`269fc42`、`e43affd` 三個本機 commit，以及本次修正與文件。

**目前適合先試用 CPU 與整張 A30；不能說 MPS 共享 GPU 已全部驗證正常。** 修改已以 `MarkJiang-maji/kjob:cgu-a30-fixes` 提供試用，尚未全平台部署。PR #1 已依使用者指示關閉、尚未合併；安裝與測試不需要 PR。試用與正式上線指令見 [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)。

## 本次發現與處理

| 發現 | 處理 |
|---|---|
| 試用文件仍說 Python 3.8 不能安裝，與 pyproject.toml 不一致 | 改為 Python 3.8 以上；wheel 實測環境為 Python 3.9.13，另外確認原始碼通過 Python 3.8 語法解析 |
| GitHub fork 的 main 只有原始版本 | 試用指定 cgu-a30-fixes 分支，保留 wheel 安裝流程；main 等老師審核後再使用 |
| root 工作在 gpunode051 卡住，與「MPS server 一定是 root」的假設不符 | 唯讀檢查確認 server 以 UID 1000 執行，root 客戶端 CUDA 初始化卡住；修正文案，推薦整卡，沒有重啟共用 daemon |
| backoffLimit: 0 讓舊 operator 刪除失敗 Pod，狀態更新前又重建，CPU 腳本執行兩次 | 省略 backoffLimit，保留 replica restartPolicy: Never；實測退出碼 3、OOM 都只執行一次，Failed Pod 保留 |
| 指定 gpunode052 的 GPU 工作一直 Pending | 實際原因是該節點 GPU 已滿；不修改別人的工作，改在已有空卡的節點驗證等候接續，文件建議一般省略 -w |

失敗重跑的判斷依據包含現場事件、兩次 start 的日誌，以及目前 operator 的 [失敗清理邏輯](https://github.com/kubeflow/training-operator/blob/855e096/pkg/controller.v1/common/job.go#L190-L246) 和 [Never 的正常失敗處理](https://github.com/kubeflow/training-operator/blob/855e096/pkg/controller.v1/pytorch/pytorchjob_controller.go#L436-L450)。修正只改新工作產生的 manifest，不升級 operator。

## 驗證結果

- 本機與 Notebook 單元測試：49 項通過。
- `mtest/kjob-test` 完整實測：11 個檢查通過，包括跨節點 CPU／NFS、CPU 看不到 GPU、群組目錄唯讀、整張 A30 CUDA、GPU 配額滿後接續、預設避開 MPS、四種提交前拒絕、刪除時停止監看／PVC 日誌、Ctrl-C 取消，以及 OOM 退出碼。
- 完整實測後發現上述失敗重跑問題，修正後另做兩個針對性檢查：退出碼 3、OOM 137 均只執行一次，保留相同 UID 的 Failed Pod；OOM 顯示 `OOMKilled`。
- wheel 安裝後實際提交 CPU 範例成功；最終 wheel 另驗證整張 A30 CUDA 工作。
- 文件裡的 Bash 範例通過語法檢查；image 建置範例沒有實際建置或發布。

測試使用既有 `mtest/kjob-test`，只清除此輪建立的工作。舊的九個 PyTorchJob、Notebook、PVC 與 notebook-sbatch 備份均保留。測試完成後恢復 Notebook 原來的停止狀態。

## 保留待後續處理

- MPS server 的 UID 切換、健康與各種 image 的相容性。這次沒有更動 daemon、device plugin 或節點。
- 既有 GPU Notebook 的 `/dev/shm` 重複掛載問題；kjob 避免自己的 MPS 工作重複掛載，不會修復原本卡住的 Notebook。
- 按 GPU 型號限制配額，以及群組／NFS 權限的服務端檢查。客戶端工具不能取代平台授權。
- 嚴格 FIFO／公平排隊、分散式 Worker、sattach、image override。現在的等候由既有 operator 重試與 Kubernetes scheduler 處理。
- 其他 namespace 的實測與正式 image 預裝。沒有宣稱本次 mtest 成功代表全平台已驗證。
