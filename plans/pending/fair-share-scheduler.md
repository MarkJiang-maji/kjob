# kjob 公平排程器（fair-share scheduler）計畫

狀態：**待核准，尚未實作**。來源：2026-11-11 會議筆記（自己寫 controller；先 namespace 之間公平、再決定 job；submit 進來就 suspend；過去一週用量越多 priority 越低；參考 Slurm；要定義 kill）。

## 現況（2026-11-11 在 .4 唯讀查證）

| 項目 | 現況 | 對設計的影響 |
|---|---|---|
| Training Operator | `kubeflow/training-operator:v1-855e096`，PyTorchJob CRD 有 `runPolicy.suspend` | 可以「先 suspend、輪到才放行」；實際行為在 Phase 0 實測 |
| Kueue / Volcano / 自訂 PriorityClass | 都沒有 | 自己寫，不會跟現有排程器衝突 |
| GPU 使用者 | Notebook 佔 31 張（另有 10 張 Pending），kjob job 6 張 | 只排 job 只管到少數 GPU；Notebook 的用量要不要算進 priority 是關鍵決定 |
| kjob 狀態服務 | `kjob-system/kjob-status` 已在跑，`kjob/cluster.py` 的 `build_snapshot()` 算好每個節點的 GPU 用量和每個 GPU workload | 排程器直接重用，判斷「放不放得下」 |
| 計費 metric | `cgu_namespace_billable_gpu_cards{user_namespace}`，Prometheus 沒有 PVC，而且會算 Pending Pod | **不拿來當用量來源**：重啟會掉資料，排隊時間也會被當成用量 |
| kjob job 標記 | label `app.kubernetes.io/managed-by=kjob`、annotation `kjob/submitted` | 排序用的送出時間已經有 |
| 時間限制 `-t` | Pod 的 `activeDeadlineSeconds`，從 Pod 開始算 | 排隊時間不會吃掉使用者的時間限制 |

## 設計

```text
ksub ──建立 PyTorchJob（suspend: true, label kjob/queue=fair, kjob/priority）──▶ API server
                                                                                   │
kjob-system/kjob-scheduler（每 30 秒）                                              │
  1. 讀所有 kjob/queue=fair 且 suspend 中的 job ◀───────────────────────────────────┘
  2. build_snapshot()：每種 GPU 的空位、各 namespace 正在用的 GPU
  3. 更新用量帳本（ConfigMap kjob-scheduler-usage，半衰期衰減）
  4. namespace priority：F = 2^(-U/S)，高的先挑
  5. namespace 內：kjob/priority，再看 kjob/submitted
  6. 放得下就 suspend=false；放不下就跳過（backfill）
  7. 在 job 上寫 annotation：排第幾、為什麼在等
```

- **程式位置：** `kjob/scheduler.py`（迴圈與 API 讀寫）、`kjob/fairshare.py`（純函式：用量衰減、priority、排序、放不放得下、選要砍的 job）、`deploy/scheduler.yaml`、`tests/test_fairshare.py`。
- **部署：** 和狀態服務一樣在 `kjob-system`，從固定 commit 安裝，有 `--dry-run`（只記錄決定，不改 job）。
- **用量：** 排程器每次迴圈把 Running 的 GPU Pod 換算成卡·小時，乘上 GPU 種類權重（預設都是 1），存進 ConfigMap，依半衰期衰減。重啟不會掉資料，Pending 不算。
- **相容：** `ksub` 只有偵測到排程器的心跳（ConfigMap 時間戳）才送 suspend，所以沒裝排程器的叢集（例如 .6）行為不變。CPU-only job（`-G 0`）不排隊。

## Phases

### Phase 0：實測 suspend（風險最高，先做）
Verified when：在 .4 測試 namespace 證實三件事都成立。
Files to touch：無（只建暫時物件，測完刪除）

#### Step 0.1：suspend／放行／再 suspend（Size: XS）
- Acceptance criteria：suspend 中沒有 Pod，quota 不被佔用；改成 false 後 Pod 正常跑；再改回 true 時 Pod 被刪、job 狀態可再放行。
- Verification command：`kubectl get pytorchjob,pods -n <test-ns> -w` 加上 conditions 紀錄
- Files likely touched：無
- Test seams：live cluster
- Blocked by：none

### Phase 1：排隊與公平排序（不砍 job）
Verified when：在 .4 dry-run 3 天，決定紀錄合理；開啟後用兩個測試 namespace 驗證「用量少的先放行」，而且放行的 job 不會因 GPU 不足卡 Pending 超過 2 分鐘。
Files to touch：`kjob/fairshare.py`、`kjob/scheduler.py`、`deploy/scheduler.yaml`、`kjob/core.py`、`kjob/cli.py`、`tests/`、`docs/ADMIN.md`、`docs/USER_GUIDE.md`、`kjob/skills/kjob-jobs/SKILL.md`

#### Step 1.1：fairshare 純函式與單元測試（Size: S）
- Acceptance criteria：衰減、priority、排序、backfill、放不放得下（GPU 種類、節點、taint、namespace quota）都有測試。
- Verification command：`python -m pytest tests/test_fairshare.py`
- Files likely touched：`kjob/fairshare.py`、`tests/test_fairshare.py`
- Test seams：純函式，輸入是 snapshot dict 和 job dict
- Blocked by：0.1

#### Step 1.2：排程迴圈與用量帳本（Size: M）
- Acceptance criteria：dry-run 只寫 log；正式模式只改 `kjob/queue=fair` 的 job；帳本存在 ConfigMap，重啟後延續。
- Verification command：`python -m pytest tests/test_scheduler.py`（假 API）＋ .4 dry-run log
- Files likely touched：`kjob/scheduler.py`、`kjob/cli.py`（`kjob scheduler` 子指令）、`tests/test_scheduler.py`
- Test seams：以假 CustomObjectsApi／CoreV1Api 注入
- Blocked by：1.1

#### Step 1.3：部署檔（Size: S）
- Acceptance criteria：最小權限（cluster-wide list nodes/pods、get/list/patch pytorchjobs、自己的兩個 ConfigMap）；`kubectl delete -f` 能完整移除。
- Verification command：`kubectl apply --dry-run=server -f deploy/scheduler.yaml`
- Files likely touched：`deploy/scheduler.yaml`、`docs/ADMIN.md`
- Test seams：server dry-run
- Blocked by：1.2

#### Step 1.4：kjob 指令端（Size: M）
- Acceptance criteria：`ksub --priority low|normal|high`；有心跳才送 suspend；`kqueue`／`kshow` 顯示 QUEUED、排第幾、等待原因；`ksrun`／`kwait` 排隊時印出原因。
- Verification command：`python -m pytest tests/`
- Files likely touched：`kjob/core.py`、`kjob/cli.py`、`docs/USER_GUIDE.md`、`kjob/skills/kjob-jobs/SKILL.md`
- Test seams：既有 CLI 測試
- Blocked by：1.2

#### Step 1.5：.4 上線（Size: S）
- Acceptance criteria：dry-run 3 天 → 開啟；兩個測試 namespace 驗證順序；回復方式是 `kubectl delete -f deploy/scheduler.yaml` 加上把所有 suspend 中的 job 放行。
- Verification command：驗收腳本 `scripts/verify-scheduler.sh`（列出 queue、帳本、最近決定）
- Files likely touched：`scripts/verify-scheduler.sh`
- Test seams：live cluster
- Blocked by：1.3、1.4

### Phase 2：砍 job（preemption）
Verified when：測試 namespace 中，低 priority namespace 的 job 被放回佇列，高 priority namespace 的 job 拿到 GPU；保護時間內的 job 不會被砍。
Files to touch：`kjob/fairshare.py`、`kjob/scheduler.py`、`kjob/cli.py`、docs、skill

#### Step 2.1：選擇被砍對象（Size: S）
- Acceptance criteria：只從 priority 最低的 namespace 挑；先砍 `kjob/priority` 低的；同級隨機；跑不到保護時間的不砍。
- Verification command：`python -m pytest tests/test_fairshare.py -k preempt`
- Blocked by：1.5

#### Step 2.2：執行與通知（Size: M）
- Acceptance criteria：被砍的 job 改回 suspend（放回佇列）；寫 Event 和 annotation；`kqueue`／`kacct` 看得到「被排程器放回佇列」。
- Verification command：`python -m pytest tests/` ＋ .4 測試 namespace 實測
- Blocked by：2.1

## Decision log
| Decision | Chosen option | Why |
| --- | --- | --- |
| 用量來源 | 排程器自己記，存 ConfigMap | Prometheus 沒 PVC、會算 Pending |
| 放不放得下 | 重用 `cluster.build_snapshot()` | 已處理 GPU 種類、MPS、taint，`ksinfo` 用同一份資料 |
| 沒裝排程器時 | `ksub` 照舊不 suspend | 避免 job 永遠卡住；.6 不受影響 |
| 被砍的 job | 待 gate 決定（建議放回佇列） | |

## Not yet specified
- Phase 3（強制所有 PyTorchJob 先 suspend 的 admission webhook，避免用 kubectl 繞過）：fog，Phase 2 完成後再研究。
- 大 job（例如 8 卡）可能一直被小 job backfill 搶走：Phase 1 先觀察 dry-run 資料，必要時加 age 因子或預留。

## Always / Never
- Always：先 dry-run；只動 `kjob/queue=fair` 的 job；每個 Phase 驗收過才往下；在 `fair-scheduler` 分支開發。
- Never：不砍、不排 Notebook；不動非 kjob 建立的 PyTorchJob；不改 Training Operator；沒核准不 push 到 GitHub `main`、不動 .6。

## Progress
| Phase | Status |
| --- | --- |
| 0 實測 suspend | 未開始 |
| 1 排隊與公平排序 | 未開始 |
| 2 砍 job | 未開始 |
