# O005 根修与 Scientific Matrix v1 验证证据

**候选状态：等待用户验收。** 本记录不表示该阶段已被用户接受或完成。

## 实现版本

| 项目 | 记录 |
|---|---|
| 分支 | `codex/v3-m3-extension` |
| 实现提交 | `3332924a87bad6f4bdffc7ce21232419afb58801` |
| ORCA 资源预算 | 4 核、总内存 1024 MB、`%maxcore 192`、并发 1 |

## 离线验证

| 检查 | 结果 |
|---|---|
| 完整 pytest | 432 passed、16 skipped；跳过项要求显式启用实时 ORCA 测试 |
| `ruff check src tests` | 通过 |
| `ruff format --check src tests` | 通过，100 个文件已格式化 |
| `git diff --check` | 通过 |

## O005 与 Benchmark v1 实时 ORCA

命令：

```powershell
.\.venv\Scripts\python.exe -m bg6022.benchmark run benchmarks/v1 --live-orca --config config.toml
```

报告：`benchmarks/results/20260923_190402_765378_3332924a/`

| 验收项 | 实测结果 |
|---|---|
| Benchmark v1.1 | 28/28 cases passed |
| ORCA attempts | 9；无 ORCA 失败 |
| LLM calls / repair attempts | 0 / 0 |
| Critical assertion failures / safety violations | 0 / 0 |
| O005 Plan | 3 Steps |
| O005 两个 SP 的输入几何 | 相同初始 geometry |
| O005 三个 Result | 全部 succeeded |
| O005 ORCA attempts | 2；能差 Tool 不增加 ORCA attempt |

## Scientific Matrix v1 实时 ORCA

命令：

```powershell
.\.venv\Scripts\python.exe -m bg6022.benchmark matrix benchmarks/scientific_v1 --live-orca --config config.toml
```

报告：`benchmarks/results/20260923_185632_477248_3332924a/`

| 指标 | 实测结果 |
|---|---:|
| Enabled cells | 11 |
| Passed runs | 11/11 |
| ORCA attempts | 16 |
| LLM calls / PubChem calls / repair attempts | 0 / 0 / 0 |
| Wall time | 428.91 s |

T003 的 H2O、NH3、CO2 三个 cell 均通过 `frequency_complete` 与
`local_minimum_supported`；没有因 CO2 线性频率特征放宽断言。各 cell 的
Request/Plan 通过生产校验，Request inline XYZ 与对应 geometry 文件一致。

Matrix 报告目录包含标准 benchmark 文件，以及 `matrix.json` 和
`matrix.md` 的 Task × Scientific Object 汇总。真实 ORCA 原始数据位于
`E:\BG6022-v3-benchmark-data\bench_20260923_185632_477248_3332924a`。
`benchmarks/results/` 按仓库规则被 Git 忽略，因此报告和原始计算数据保留在
本机，不随源码提交。
