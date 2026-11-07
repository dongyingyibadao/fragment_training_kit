# 昇腾低优碎卡训练参考项目

**给其他 agent 的入口：[AGENT_START_HERE.md](AGENT_START_HERE.md)。** 这是可单独克隆和运行的指导项目，与最初的实验项目处于同级目录；不需要复制原项目、建立软链接或使用原项目的 PYTHONPATH。本目录把已在 OpenPI 上验证的训练与平台流程整理成可套用的接入样例：每卡模型/Adam 常驻 NPU，HCCL 归约，CPU 控制成员和版本，APPLY 前单卡失联后保留健康梯度并补算缺失分配。

OpenPI案例已完成64张独立低优卡、global256、无重算、20,000步；最终LIBERO为1899/2000（94.95%）。本工具包新增通用 adapter 和小回归模型，用于验证他人的代码能够通过同一训练控制流接入。支持范围与实测证据见 [05](docs/05_evidence_and_limits.md)，不把案例成功扩大成任意故障均可恢复。

**本次独立接入验证已通过：** 无会话背景的agent按目录完成真实两卡20步，以及2→1受控掉卡20步；loss从4.453125降到0.16766，两轮数值参考均通过，资源均已释放。[查看接入验收](docs/06_handoff_acceptance.md)。

**整机与碎卡混合优化：** 新增节点资源规划、每作业 1/2/4/8/16 卡、固定候选池的延迟申请与步边界接入。4×16 OpenPI 30 步短训实际提交间隔均值 **1.767 秒**；600 步小模型自动切入/定时切回及严格 CPU 数值参考通过。默认 HCCL 大梯度通信约 0.108 秒，手工分层在本轮反而更慢。功能、结果和恢复限制见 [09 混合训练](docs/09_topology_training.md)。这是有界实验验收，不代表整 pod 抢占后已能自动无损恢复。

## 目录与读取顺序

| 入口 | 解决的问题 |
|---|---|
| [PLAN.md](PLAN.md) | 本次整理与实际接入验收计划 |
| [01 架构](docs/01_architecture.md) | 一步更新、样本权重、失联、替补入组 |
| [02 接入自己的模型](docs/02_adapting_your_model.md) | 模型/数据接口、精度、随机性与梯度约束 |
| [03 平台与自动占卡](docs/03_platform_and_resources.md) | 登录、资源发现、配置、预算与自有作业清理 |
| [04 验收与性能](docs/04_validation_and_performance.md) | loss、梯度/Adam参考、正常步/故障/启动计时 |
| [05 已有证据与边界](docs/05_evidence_and_limits.md) | 已验证什么，哪些情形需要额外实现 |
| [06 本次接入验收](docs/06_handoff_acceptance.md) | 不继承对话的agent的真实短跑结果 |
| [07 独立项目与依赖](docs/07_standalone_project.md) | 目录迁移、外部输入、历史证据和验证范围 |
| [08 拓扑执行计划](docs/08_topology_execution_plan.md)、[09 混合训练](docs/09_topology_training.md) | 节点规划、多卡作业、timeout对照、候选切换与限制 |
| `adapters.py`、[模型模板](examples/model_adapter_template.py) | 可修改的模型接入点 |
| [toy代码](examples/toy_regression.py)、[数值审计](examples/audit_toy.py) | 无外部数据、CPU参考与真实NPU短训练 |
| [站点示例](config/site.example.json)、`launch.py` | prepare/verify/run/observe/cleanup |
| [OpenPI案例字段](config/openpi.case.template.json) | 把已验证案例映射到通用站点配置 |
| `vendor/`、`runtime_manifest.json` | 随包运行时源码与其SHA清单 |

## 环境条件

训练作业需要兼容的 torch/torch_npu/CANN/HCCL、与申请规格一致的可见 NPU、跨作业网络、共享文件系统和可用的 QZ 登录状态。多卡作业保留完整设备mask，每个进程选择独立逻辑设备；不能再次把可见域切成只有一张卡。当前已验证后端限定两个 QZ 空间，其他站点需移植资源后端。toy还需要NumPy；不需要OpenPI、模型下载或数据集。CPU参考可用普通CPU PyTorch；CPU检查的依赖列在 [requirements-cpu.txt](requirements-cpu.txt)，在自己的环境中安装。NPU环境需单独匹配版本，不能用这份CPU依赖文件升级共享NPU环境。

可以把本目录单独复制到共享盘其他位置。`launch.py`仅使用本项目的vendor，缺失时直接报错，不回退到原项目；prepare会冻结本轮执行代码。自己的模块通过site中的`source_dirs`提供**纯源码目录**，prepare后使用快照，不继续执行可变原文件。数据、权重、环境、私有inventory和登录文件由站点另供。

当前训练合同固定global batch256、atomic4、2–64个初始成员、原生AdamW、无梯度重算。先跑两卡20步；其他batch/optimizer/分片状态需要新的适配与验收。

## 最短可执行流程

先克隆项目，以下命令都从包含 `fragment_training_kit/` 的父目录执行；保留这个目录名即可使用 Python 模块入口。

```bash
git clone https://github.com/dongyingyibadao/fragment_training_kit.git fragment_training_kit
```

将 `fragment_training_kit/config/site.example.json` 复制到项目外，填好 Python、CANN、私有inventory、共享registry等站点路径。登录文件通过CLI传入，不写进site.json。

```bash
python fragment_training_kit/tools/verify_package.py
python -m fragment_training_kit.examples.audit_toy --cpu-oracle --steps 20 --output /shared/work/toy-cpu-oracle.json
python fragment_training_kit/launch.py prepare --config /shared/private/site.json --run-dir /shared/work/toy-low2-v1
python fragment_training_kit/launch.py verify --run-dir /shared/work/toy-low2-v1
python fragment_training_kit/launch.py run --run-dir /shared/work/toy-low2-v1 --cookie-file /shared/private/qz.cookies --watch-seconds 1200
python -m fragment_training_kit.examples.audit_toy --run-dir /shared/work/toy-low2-v1/probe --output /shared/work/toy-low2-audit.json
python fragment_training_kit/launch.py observe --run-dir /shared/work/toy-low2-v1 --cookie-file /shared/private/qz.cookies
```

上面的 `python` 使用包含torch/NumPy的解释器；NPU作业实际解释器由site.python指定。示例路径需要替换。prepare不提交任务；run会创建配置预算内的LOW/1作业并监控到结束。正常结束优先等待作业自然成功退出。需要中止或异常收尾时执行：

```bash
python fragment_training_kit/launch.py cleanup --run-dir /shared/work/toy-low2-v1 --cookie-file /shared/private/qz.cookies
```

保留该命令输出及`cleanup_receipt.json`。只有官方终态、运行/排队/预留为0才算释放。run有时间上限，超时不是训练通过；使用新run保存下一次尝试。

## 接入自己的代码

复制 `examples/model_adapter_template.py` 到自己的纯代码目录，实现模型、`prepare_samples` 和逐样本 `losses`，在site.training_config中设置 `resident_adapter="your_module:create_engine"`，并把模块所在目录放入site.source_dirs。基类按每样本1/256累积梯度，归约、裁剪和Adam仍由现有运行时执行。适配后先按 [02](docs/02_adapting_your_model.md) 与 [04](docs/04_validation_and_performance.md) 做自己的数值/短跑验收。

平台自动申请和训练入组是两个门槛。新分配的任意job不会自动进入固定成员组；当前launcher避免持续补占不能被controller接纳的新ID。受控替补运行在同一存活supervisor下；真实整作业丢失可在已支持阶段缩组继续，后续完整checkpoint新run恢复需显式规划。

底层controller的checkpoint恢复需要checkpoint和原source-spec一起绑定；此launcher目前未提供一键resume参数，接入者需单独接入并验证，不能假设prepare会自动续训。

维护者有意升级运行时代码时，执行 `tools/snapshot_runtime.py --source-root /path/to/source --refresh`，重新验收并发布新的manifest。维护工具要求显式来源，不会自动寻找原实验项目；普通训练和模型接入都不需要运行它。

随包回归测试可从本目录执行：`TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q tests`。测试只验证接口、CPU数学、冻结和平台生命周期规则，不分配NPU；真实两卡接入证据见06。

随附的toy历史证据也可完全在本目录内重新核验，命令见 [07](docs/07_standalone_project.md)。这属于保存证据的CPU审计，不是重新执行NPU训练。历史GPU/JAX引擎与LIBERO评估辅助模块不属于本独立项目的可执行入口。
