# kjob：在 Kubeflow Notebook 裡像國網一樣提交 GPU 工作

kjob 把你在 Notebook 寫好的程式，提交成叢集上的一個工作（Kubeflow PyTorchJob）。工作使用和 Notebook 相同的
image、工作目錄和共用磁碟，所以程式、資料、虛擬環境在工作裡都在同樣的路徑。輸出會即時顯示，也會存到
`~/kjob-logs/工作名稱.log`。

指令仿照國網（Slurm）：`ksub` ≈ `sbatch`、`ksrun` ≈ `srun`、`kqueue` ≈ `squeue`、`ksinfo` ≈ `sinfo`、
`kacct` ≈ `sacct`、`kcancel` ≈ `scancel`。

* Notebook 本身不用占 GPU：訓練交給工作，跑完就釋放。
* 看得到整個叢集：每種 GPU 有幾張、用了幾張、誰在用、多少人在排隊。
* 可以讓 AI 助手代你提交：內附 Claude Code 等工具可用的 skill。
* 不同機器的 GPU 不一樣（例如 V100、A5000、A30、RTX PRO 6000），用 `ksinfo` 查目前這台叢集有哪些。

## 安裝（在 Notebook 的 Terminal）

```bash
python3 -m pip install --user --upgrade https://github.com/MarkJiang-maji/kjob/archive/refs/heads/cgu-public.tar.gz
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
kinfo      # 確認安裝成功：顯示 namespace、工作會繼承的磁碟、GPU 配額
```

需要 Python 3.8 以上。安裝在 `~/.local`（工作磁碟），Notebook 重開仍然在。若在虛擬環境內，拿掉 `--user`。

## 五分鐘上手

```bash
ksinfo                                    # 1. 看哪種 GPU 有空
ksub -J test -G 1 -g v100 -c 4 --mem 16Gi -t 2:00:00 train.py --epochs 10   # 2. 提交
kqueue                                    # 3. 看自己的工作
klogs test -f                             # 4. 即時看輸出（Ctrl-C 只停止觀看）
kacct                                     # 5. 看歷史紀錄與結束代碼
```

完整新手教學請看 [TRIAL_GUIDE.md](TRIAL_GUIDE.md)，簡報版在 [docs/kjob-tutorial.pptx](docs/kjob-tutorial.pptx)。

## 指令一覽

| 國網 / Slurm | kjob | 用途 |
|---|---|---|
| `sbatch` | `ksub [選項] 腳本 [參數]` | 提交後立即返回 |
| `srun` | `ksrun [選項] 腳本 [參數]` | 提交並即時顯示輸出，結束後自動刪除工作；Ctrl-C 取消 |
| `squeue -u $USER` | `kqueue` | 自己 namespace 的工作 |
| `squeue` | `kqueue -A` | 整個叢集所有使用者的 GPU 工作與排隊原因 |
| `sinfo` | `ksinfo [-g 型號]` | 每種 GPU、每個節點的總數／使用中／空閒／排隊 |
| `scontrol show job` | `kshow 工作名稱` | 單一工作的資源、限制、狀態、事件 |
| — | `klogs 工作名稱 [-f]` | 輸出（工作刪除後改讀日誌檔） |
| — | `kwait 工作名稱` | 等工作結束並回傳結束代碼 |
| `sacct` | `kacct` | 自己提交過的工作、結束代碼、執行時間 |
| `scancel` | `kcancel 工作名稱` | 取消工作 |
| — | `kclean` | 刪除已結束的工作物件（日誌保留） |
| — | `kinfo` | 工作會繼承的 image、磁碟、配額 |

常用提交選項：`-G` GPU 數、`-g` GPU 型號、`-c` CPU、`--mem` 記憶體、`--shm` 共享記憶體、`-t` 時間上限、
`-J` 名稱、`-w` 指定節點。腳本開頭也可以寫 `#SBATCH` 或 `#KJOB` 設定。全部選項與說明見
[docs/USER_GUIDE.md](docs/USER_GUIDE.md)。

## 讓 AI 助手幫你

安裝後執行：

```bash
kjob install-skills          # 安裝到 ~/.claude/skills（Claude Code）；其他工具用 --dir 指定
```

之後可以直接跟 AI 說「幫我用 kjob 把 train.py 交到有空的 GPU 上跑，記憶體 32G」。還沒安裝 kjob 時，可以請 AI
閱讀 [kjob/skills/kjob-install/SKILL.md](kjob/skills/kjob-install/SKILL.md) 照步驟安裝。

## 文件

| 文件 | 對象 |
|---|---|
| [TRIAL_GUIDE.md](TRIAL_GUIDE.md) | 第一次使用：安裝、提交 GPU 範例、查看結果 |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | 全部指令與選項、排隊與錯誤排解、Python API |
| [docs/kjob-tutorial.pptx](docs/kjob-tutorial.pptx) | 教學簡報 |
| [docs/ADMIN.md](docs/ADMIN.md) | 叢集管理員：權限需求、叢集狀態服務、GPU 標籤 |
| [kjob/skills/](kjob/skills/) | AI 助手 skill：安裝 kjob、提交與監看工作 |

## 限制

* 工作沿用 Notebook 的 image，不能另外指定 image；只在 Notebook 裡 `pip install`（非 `--user`）的套件，工作看不到。
* 腳本、資料、輸出要放在共用磁碟（`/home/jovyan` 或群組資料夾）；Notebook 的 `/tmp` 工作看不到。
* 排隊由 Kubernetes 排程器處理，不是嚴格先到先得；namespace 的 GPU 配額包含 Notebook 本身占用的 GPU。
* 不支援 job array、相依工作（`--dependency`）、`sattach`。
