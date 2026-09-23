# R1 修复复评候选验收记录

状态：`implementation_and_live_verified_awaiting_user_acceptance`。代码候选提交：`303a4fbba22bac421dbd0cab668d2918885e764f`，父提交为复评基线 `d67a24976dcfd9c03865bb5b053d2b334e9094b2`，分支 `codex/structural-rebuild`。本记录只覆盖 `R1_REPAIR_REASSESSMENT.md` 的收尾范围，不进入 R2，也不代表用户已验收。历史文件 [`R1-execution-evidence.md`](R1-execution-evidence.md) 保留不变。

## 修复结果

- 恢复按最大 attempt 的生命周期记录与目录排序，再核验当前 Step 指纹、Result 身份、输入绑定及 Artifact。最新失败会恢复为失败，取消/中断保持停止，`needs_input` 恢复为等待；较新的未知 attempt 阻断旧成功。新修订可以取代旧澄清/已结束尝试，旧失败不会压过较新的已验证成功。
- ORCA、PubChem 和几何适配器不再自行创建 attempt。`Tool.execute` 和适配器只接受 `Agent.advance` 在 Run owner、预算准入后创建的活动上下文；上下文同时绑定 owner、预算保留、Plan revision 与 Step 指纹。预算耗尽、busy owner、取消后的陈旧对象及未授权的直接入口都以零次适配器/runner 调用结束。
- CLI 确认队列保存最近实际展示的 Run/指纹；未展示的新预览不能替换已展示的授权对象。另将忙碌取消答复改为不声称跨 owner 的取消信号已送达。
- 按复评点格式化了 `agent.py`、`tools/orca.py`、`geometry_angle_tool.py`、`test_r1_closeout_repairs.py`，并把直接适配器单测迁至 owner/预算准入测试网关。

## 工程与离线验证

| 检查 | 结果 |
|---|---|
| `python -m pytest -m "not live_orca" --disable-socket -q` | 406 passed、11 skipped、5 deselected，退出码 0 |
| `python -m pytest -m windows_process -q` | 2 passed、420 deselected，退出码 0 |
| `ruff check .` / `ruff format --check .` | 通过；70 个 Python 文件格式检查通过（Ruff 0.15.1） |
| `python -m compileall -q src` / `git diff --check` | 通过 |
| `uv lock --check` | 通过；报告 23 个锁定包 |
| `uv build --out-dir ...` | sdist、wheel 均构建成功（uv 0.12.18） |
| 包外干净安装 | Python 3.11.4 外部环境按冻结锁安装依赖，再安装 wheel；`python -I` 导入路径位于外部 `install-env\Lib\site-packages`，CLI `--help` 退出码 0 |

## 当前候选真实 ORCA 小计算

使用包外干净安装的 `bg6022==0.1.0` CLI。先执行不带 `--execute` 的 SP 预览：退出码 0，明确输出 `preview: no calculation executed`，显示 4 cores、总内存 1024 MB、`%maxcore 192`、并发 1。随后仅执行一次显式授权的 `run-tool single_point ... --execute`；该命令通过 `Agent.execute_plan → Agent.advance → RunOwner/attempt admission → Tool → 既有 ORCA runner`。

| 事实 | 结果 |
|---|---|
| 计算 | 水分子 SP，3 atoms，`r2SCAN-3c` / gas，charge 0 / multiplicity 1 |
| ORCA | 6.1.1，`E:\ORCA\orca.exe` |
| Run | `run_1afa8b8fc60543d691d76778e7d262d0`，Run 与 Result 均为 `succeeded` |
| 启动/预算 | Run 内 1 个计算 attempt / ORCA PID（12620），无第二次 attempt；`attempt_counts.compute = 1` |
| 结果 | `-76.417246084177 Eh`；exit code 0、normal termination、SCF converged、有限能量、input hashes match、process tree empty 均为 true |
| 资源 | 4 cores、1024 MB total、`%maxcore 192`、并发 1 |
| 随后查询 | 包外安装的 `show-run` 只读成功；查询后仍为 1 个 attempt 目录，未再次执行 ORCA |

原始 Run、Result、输入/输出和命令输出保存在仓库外：`E:\BG6022-r1-reassessment-20260923\`。配置快照也保存在该目录；没有把原始科学日志或临时配置提交进仓库。

关键 SHA-256：

| 对象 | SHA-256 |
|---|---|
| 真实计算使用的包外 wheel（提交前从已验证工作树构建；该源码树未再修改并提交为本代码提交） | `AD8371F380E616941621ACBE74E3A229070C8A570686F3CD427F88D4D14E5F62` |
| 代码提交后重新构建并安装的 wheel | `6143F3451C8E505A126E87953C3C7FC3FC29B2383357805D74FF7D8EB1C1D887` |
| 冻结依赖 `uv.lock` | `4EBCFF69853F57680C368E061B57E1EA0A1CD2171EBFB8E426C95A5B40128BF3` |
| ORCA 可执行文件 | `8D6B51BF4093C967DBED997CC651F0212B8F94313EE77EA56F548F000672C42F` |
| 临时配置快照 | `30782EFBB711F84E86EDC38A403E680C180EA9972BC2314B25D1C085CDEC4A6F` |
| 输入 `input.inp` / 几何快照 | `57BA5323CB7E48A2812178CE53803FC1E09A48ECCB2C8F5E3AA6B23A11DAB299` / `C4DD083DE426421A719338BD0F0E17DC1057A88982E8CF245FB115556525FFD2` |
| Result / ORCA stdout / Run checkpoint | `8BCC6E4C45AFA709C5EC07CECFF3B626048DCD9463D922A3443EDC9CCEC980A3` / `4413118EE60595A6B8E3E678034733EB7D3C845152937759C01C75CB204217D2` / `EF44BCCDC7E213CED16921ECF351517E762207F014025B53EB90A5185A856DD1` |

随包提示词在本候选中未改动；源码 SHA-256：

| 提示词 | SHA-256 |
|---|---|
| `intake.md` | `CC7D326FAB9F8D3067EEE8E38C1E4D5C7BD80FD96BB676DE02C400F163BFA8AB` |
| `planner.md` | `C88DC578545B2EBBC2ADD3B80FB73D70183ED56934E8A9FED984671B5B8DEEE7` |
| `answer.md` | `0CBD124A53B81F44481B9F3E4FBA88F805BEFAEA80B3A33073F196914942A353` |
| `repair.md` | `B5220C281EC8936AE2E79462BD63F5F16E1DC00E0686BA1D593B62C2974FE6D2` |

## 复评项与交接

复评提出的恢复排序/状态、直接 Tool 旁路、展示预览绑定、busy cancel 措辞、Ruff 与候选证据更新均已由本提交和上表覆盖。没有改数据库 schema、长期领域对象、ORCA parser/scientific checks、资源预算或 R2 内容；真实 LLM/PubChem 未运行，也不是本次复评的强制门。

实现、离线/Windows 检查、包外 wheel 和真实 ORCA 门现已通过；R1 仍等待用户验收。验收前不标记 R1 complete，不冻结后续阶段，不开始 R2。
