# 昇腾低优碎卡训练参考项目

**当前策略入口：[昇腾低优训练使用手册](docs/12_current_ascend_strategy.md)。给其他 agent 的入口：[AGENT_START_HERE.md](AGENT_START_HERE.md)。** 这是可单独克隆和运行的指导项目，与最初的实验项目处于同级目录；不需要复制原项目、建立软链接或使用原项目的 PYTHONPATH。当前可复用组合为：按空闲节点规划 LOW/1 多卡/单卡作业、每卡模型与 Adam 常驻 NPU、HCCL 梯度 SUM、分阶段掉卡恢复，以及累计 16 张后的完整状态同步补入。

OpenPI案例已完成64张独立低优卡、global256、无重算、20,000步；最终LIBERO为1899/2000（94.95%）。本工具包新增通用 adapter 和小回归模型，用于验证他人的代码能够通过同一训练控制流接入。支持范围与实测证据见 [05](docs/05_evidence_and_limits.md)，不把案例成功扩大成任意故障均可恢复。

**历史独立接入验证已通过：** 无会话背景的agent按目录完成真实两卡20步，以及2→1受控掉卡20步；loss从4.453125降到0.16766，两轮数值参考均通过，资源均已释放。[查看接入验收](docs/06_handoff_acceptance.md)。

**按空闲资源申请：** 使用官方节点视图，核对 Ready、NPU、CPU、内存和真实 LOW 规格，优先较完整的节点及合法的 16/8/4/2/1 卡作业；容量不足时在预算内排队单卡。初始布局由 `topology.py prepare` 规划冻结，训练中的候选由 `launch.py run` 持续规划。计划不保证调度到指定物理节点。完整使用流程与满组/部分初始组的区别见 [当前手册](docs/12_current_ascend_strategy.md)。

**掉卡与补卡（显式启用）：** 同时设置 `resident_pod_recovery=true`、`resident_pod_refill=true`。不足 16 张候选只保留资源；凑齐 16 张并冷加载后，在 COMMIT 边界同步模型、Adam 与步数，核验后整批接入。真实 OpenPI 完成 64→48→64 的 150 步短跑，补入后约 **3.4 秒/步**，一次约 20.95 GB 完整状态同步暂停 **321.6 秒**；该暂停尚未优化。toy 连续两次补卡及 15 张不加载对照通过独立 CPU 数值审计，独立副本 625 项测试通过。见 [11](docs/11_batched_refill.md)。这些短跑不替代历史 20,000 步或 LIBERO 证据；保持同步训练，不使用旧权重异步参与。

## 目录与读取顺序

| 入口 | 解决的问题 |
|---|---|
| [12 当前昇腾低优策略](docs/12_current_ascend_strategy.md) | 推荐组合、空闲资源申请、配置选择、完整命令、故障与补卡、推广边界 |
| [PLAN.md](PLAN.md) | 本次整理与实际接入验收计划 |
| [01 架构](docs/01_architecture.md) | 一步更新、样本权重、失联、替补入组 |
| [02 接入自己的模型](docs/02_adapting_your_model.md) | 模型/数据接口、精度、随机性与梯度约束 |
| [03 平台与自动占卡](docs/03_platform_and_resources.md) | 登录、资源发现、配置、预算与自有作业清理 |
| [04 验收与性能](docs/04_validation_and_performance.md) | loss、梯度/Adam参考、正常步/故障/启动计时 |
| [05 已有证据与边界](docs/05_evidence_and_limits.md) | 已验证什么，哪些情形需要额外实现 |
| [06 本次接入验收](docs/06_handoff_acceptance.md) | 不继承对话的agent的真实短跑结果 |
| [07 独立项目与依赖](docs/07_standalone_project.md) | 目录迁移、外部输入、历史证据和验证范围 |
| [08 拓扑执行计划](docs/08_topology_execution_plan.md)、[09 混合训练](docs/09_topology_training.md) | 节点规划、多卡作业、timeout对照、候选切换与限制 |
| [10 整组与连续掉卡恢复](docs/10_pod_loss_recovery.md) | 显式启用的分阶段缩组、部分 APPLY 检查、checkpoint 换源与验收边界 |
| [11 累计16张后在线补卡](docs/11_batched_refill.md) | 动态身份、无模型预留、成批加载、完整模型/Adam同步与更新边界接入 |
| `adapters.py`、[模型模板](examples/model_adapter_template.py) | 可修改的模型接入点 |
| [toy代码](examples/toy_regression.py)、[数值审计](examples/audit_toy.py) | 无外部数据、CPU参考与真实NPU短训练 |
| [站点示例](config/site.example.json)、`launch.py` | prepare/verify/run/observe/cleanup |
| [完整初始组＋动态补卡](config/site.low_priority.example.json)、[48 卡初始组](config/site.refill.example.json) | 当前策略的两种起步配置；不含真实站点信息 |
| [OpenPI案例字段](config/openpi.case.template.json) | 把已验证案例映射到通用站点配置 |
| `vendor/`、`runtime_manifest.json` | 随包运行时源码与其SHA清单 |

## 环境条件

训练作业需要兼容的 torch/torch_npu/CANN/HCCL、与申请规格一致的可见 NPU、跨作业网络、共享文件系统和可用的 QZ 登录状态。多卡作业保留完整设备mask，每个进程选择独立逻辑设备；不能再次把可见域切成只有一张卡。当前已验证后端限定两个 QZ 空间，其他站点需移植资源后端。toy还需要NumPy；不需要OpenPI、模型下载或数据集。CPU参考可用普通CPU PyTorch；CPU检查的依赖列在 [requirements-cpu.txt](requirements-cpu.txt)，在自己的环境中安装。NPU环境需单独匹配版本，不能用这份CPU依赖文件升级共享NPU环境。

可以把本目录单独复制到共享盘其他位置。`launch.py`仅使用本项目的vendor，缺失时直接报错，不回退到原项目；prepare会冻结本轮执行代码。自己的模块通过site中的`source_dirs`提供**纯源码目录**，prepare后使用快照，不继续执行可变原文件。数据、权重、环境、私有inventory和登录文件由站点另供。

当前训练合同固定global batch256、atomic4、2–64个初始成员、原生AdamW、无梯度重算。先跑两卡20步；其他batch/optimizer/分片状态需要新的适配与验收。

## 最短可执行流程

先克隆项目，以下命令都从包含 `fragment_training_kit/` 的父目录执行；保留这个目录名即可使用 Python 模块入口。

```bash
git clone --branch feature/batched-refill https://github.com/dongyingyibadao/fragment_training_kit.git fragment_training_kit
```

当前组合位于 `feature/batched-refill`；默认分支的历史版本不一定包含它。先用两卡样例验证站点和模型：将 `fragment_training_kit/config/site.example.json` 复制到项目外，填好 Python、CANN、私有inventory、共享registry等站点路径。登录文件通过CLI传入，不写进site.json。完成后按 [当前手册](docs/12_current_ascend_strategy.md) 启用节点规划、整组恢复和动态补卡。

```bash
python fragment_training_kit/tools/verify_package.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m fragment_training_kit.examples.audit_toy --cpu-oracle --steps 20 --output /shared/work/toy-cpu-oracle.json
python fragment_training_kit/launch.py prepare --config /shared/private/site.json --run-dir /shared/work/toy-low2-v1
python fragment_training_kit/launch.py verify --run-dir /shared/work/toy-low2-v1
python fragment_training_kit/launch.py run --run-dir /shared/work/toy-low2-v1 --cookie-file /shared/private/qz.cookies --watch-seconds 1200
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m fragment_training_kit.examples.audit_toy --run-dir /shared/work/toy-low2-v1/probe --output /shared/work/toy-low2-audit.json
python fragment_training_kit/launch.py observe --run-dir /shared/work/toy-low2-v1 --cookie-file /shared/private/qz.cookies
```

上面的 `python` 使用包含torch/NumPy的解释器；NPU作业实际解释器由site.python指定。示例路径需要替换。prepare不提交任务；run会创建配置预算内的LOW/1作业并监控到结束。正常结束优先等待作业自然成功退出。需要中止或异常收尾时执行：

```bash
python fragment_training_kit/launch.py cleanup --run-dir /shared/work/toy-low2-v1 --cookie-file /shared/private/qz.cookies
```

保留该命令输出及`cleanup_receipt.json`。只有官方终态、运行/排队/预留为0才算释放。run有时间上限，超时不是训练通过；使用新run保存下一次尝试。

## 接入自己的代码

复制 `examples/model_adapter_template.py` 到自己的纯代码目录，实现模型、`prepare_samples` 和逐样本 `losses`，在site.training_config中设置 `resident_adapter="your_module:create_engine"`，并把模块所在目录放入site.source_dirs。基类按每样本1/256累积梯度，归约、裁剪和Adam仍由现有运行时执行。适配后先按 [02](docs/02_adapting_your_model.md) 与 [04](docs/04_validation_and_performance.md) 做自己的数值/短跑验收。

平台自动申请和训练入组是两个门槛。默认路径仍使用固定成员；显式启用 [16卡成批补卡](docs/11_batched_refill.md) 后，launcher 可持续申请动态成员，累计16张才冷加载，完整同步模型/Adam/step并确认新组后接入。候选失败只延后补卡，健康组继续训练；同时活动、预留、排队和未知提交合计不能超过共享物理预算。完整checkpoint新run恢复仍需显式规划。

底层controller的checkpoint恢复需要checkpoint和原source-spec一起绑定；此launcher目前未提供一键resume参数，接入者需单独接入并验证，不能假设prepare会自动续训。

维护者有意升级运行时代码时，执行 `tools/snapshot_runtime.py --source-root /path/to/source --refresh`，重新验收并发布新的manifest。维护工具要求显式来源，不会自动寻找原实验项目；普通训练和模型接入都不需要运行它。

随包回归测试可从本目录执行：`TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q tests`。测试只验证接口、CPU数学、冻结和平台生命周期规则，不分配NPU；真实两卡接入证据见06。

随附的toy历史证据也可完全在本目录内重新核验，命令见 [07](docs/07_standalone_project.md)。这属于保存证据的CPU审计，不是重新执行NPU训练。历史GPU/JAX引擎与LIBERO评估辅助模块不属于本独立项目的可执行入口。
