# 将自己的训练代码接入

先运行随包的 `examples/toy_regression.py`，它不依赖 OpenPI、权重或数据下载。接入接口是配置中的 `resident_adapter = "你的模块:工厂函数"`。工厂接收 `config, device_index=0`，返回一个 engine；`adapters.py` 定义运行时所需接口，`examples/model_adapter_template.py` 给出可修改的模板。

engine 负责模型构造、样本定位、预处理、前反向和设备同步。controller 继续负责分配、HCCL、APPLY、原生 AdamW 和完整状态；不要在 adapter 里再创建 DDP、执行 AllReduce 或调用 optimizer.step。

自定义adapter还要在training_config明确填写 `resident_rng_scheme` 与 `resident_precision_contract`，分别描述样本/模型随机数身份规则和实际参数/moments精度；`resident_data_workers=0`，由adapter自行组织纯CPU读取/预取。其他模型不能沿用OpenPI的diffusion噪声描述。

## 接口需要提供什么

| 成员 | 用途 |
|---|---|
| `model` | eager torch 模型，参数和 buffers 都可完整保存/原位恢复 |
| `parameters` | 稳定名称到全部 active Parameter 的映射，所有成员的名称/顺序/dtype 一致 |
| `config`、`torch`、`torch_device` | 调度与设备信息；设备必须使用平台分配的逻辑 `npu:0` |
| `_sync()` | 等待本设备实际计算完成，计时和状态操作使用 |
| `checkpoint_call_count` | 重算调用计数，本路径要求为 0 |
| `backward_units(base_step, units, clear)` | 按确定性单元累积原生局部梯度，返回 loss贡献、样本数和阶段计时 |

实际接口检查与可运行实现以源码为准。`clear=True` 清空本卡梯度开始一步；`clear=False` 必须保留旧梯度，只追加丢失卡样本。所有 active 参数必须有同 dtype 的梯度；条件分支、稀疏梯度、未使用参数、动态 Parameter 等需要单独适配。

## 确定性样本和随机数

每个单元编号 0–63，含4个样本。位置可按 `base_step * 256 + unit_id * 4 + offset` 定义。位置映射到数据集时，要把 epoch、样本索引和 seed 一并确定；不要使用 rank 作为决定内容的随机种子。重试不能消耗下一步样本或改变噪声。

预处理增强、dropout、diffusion noise/time 都属于训练输入合同。使用按样本位置生成的显式 RNG；若有模型内部随机算子，需要保证该样本换到另一张卡重算时含义相同。现有运行时拒绝训练态的非零 dropout。BatchNorm 等会修改 buffer 的前向也不能直接套用：在重放中修改 buffer 可能破坏健康状态保留，需要明确同步/恢复方法。toy 使用不含这些状态的线性模型。

## 梯度与精度

模型返回 loss 向量时取对应样本之和除256；返回 microbatch 均值时乘实际 microbatch样本数/256，随后 backward。不要在 adapter 内做按 world_size 的平均、梯度裁剪、优化器更新或通信。裁剪在完整 SUM 后执行一次。

adapter 需要与既有学习率、AdamW betas/eps/weight_decay 合同兼容；默认训练器并不是任意优化器框架。AMP/GradScaler、FSDP、ZeRO、FP32-master、稀疏优化器等需要扩展状态/恢复/数值验收，不能只替换模型类就假定支持。

## 接入顺序

1. 单卡测试模型在 NPU 上能前后向，关闭重算；记录 dtype、active参数数量和显存。
2. 用 CPU 小模型 oracle 对照“完整256样本梯度”和“2/3卡、非均匀分配、丢失单元补算之和”。
3. 用本目录真实两卡短跑确认相同最终模型/Adam状态，loss下降和所有梯度有限。
4. 对自己的实际模型做少步无注入、再做一次 APPLY 前失联和替补入组测试。
5. 按相同数据、精度、global batch、优化器比较固定集群与碎卡路径，扩大卡数并做长跑。

OpenPI 默认分支仍可使用：省略 `resident_adapter` 并提供原训练配置、已适配的 upstream 环境、权重、tokenizer与数据。通用 toy 包不包含 OpenPI 依赖、模型和数据；它们由使用者显式提供。OpenPI 的模型迁移优化属于案例，不属于通用运行时自动提供的能力。

字段映射示例见 [OpenPI案例配置](../config/openpi.case.template.json)：将该对象放到site.training_config，site.steps与target_steps对齐，按环境填入共享路径。它是从历史配置提炼的接入模板，不是本轮重新完整训练的证据。base_params目录需包含model.safetensors及对应assets；该案例要求mean/std的pi05_libero_lerobot，不能随意换成horizon或归一化不同的pi05_libero。

`openpi_root`需要指向你已经固定版本的upstream源码。将纯源码根目录显式放入 `source_dirs`，且 `openpi_root` 与该项完全匹配时，prepare 会复制源码并将模型根目录重绑到本 run 的冻结副本。未显式包含的外部源码不会自动冻结；第三方依赖、数据、权重版本仍需自己锁定。不要填上一次 run 的 runtime。OpenPI 建议启用 `isolate_model_cache=true`，按平台作业隔离 HF dataset 缓存，减少跨节点并发生成同一缓存的问题。

选择当前动态补卡策略时，先读 [统一使用手册](12_current_ascend_strategy.md)。用 OpenPI 案例对象替换 toy 的 `training_config` 后，再加回 `resident_pod_recovery=true`、`resident_pod_refill=true`；不能保留 toy adapter 或高学习率。通用 adapter 读取 `worker_microbatch_size`，原生 OpenPI 读取 `microbatch_size`，两者不可混淆。
