# 独立项目与依赖边界

本项目目录为 `fragment_training_kit`，与最初的 `Steadfast_as_the_Jade_Lotus` 实验目录同级。复制或克隆本项目即可，不需要原目录存在，不建立指向它的软链接。GitHub仓库也只包含本目录。

```text
your-shared-workspace/
├── fragment_training_kit/         独立源码、指导、模板和轻量证据
├── your-private-site/             自己的登录文件、inventory、站点配置
├── your-runs/                     自己的新运行及checkpoint
└── your-model-code/               可选：自己的模型adapter源码
```

## 哪些东西由本项目提供

| 内容 | 实际来源 | 是否需要原实验目录 |
|---|---|---|
| 成员控制、梯度保护、HCCL SUM、NPU Adam、故障补算 | `vendor/jade_lotus/` 与 `vendor/scripts/` | 否 |
| 自动占卡、共享预算、查询及清理 | 本项目launcher及随附QZ后端 | 否 |
| 通用adapter、toy数据、小模型和CPU数值参考 | `adapters.py`、`examples/` | 否 |
| 模型接入指导、配置模板、历史验收证据 | `docs/`、`config/`、`evidence/`、`validation/` | 否 |
| Python、torch_npu、CANN、HCCL、驱动、镜像与共享网络 | 使用者自己的站点环境 | 显式外部环境，不限于原机器 |
| 登录状态、inventory、registry、作业权限 | 使用者自己的私有站点配置 | 显式外部输入，不随源码发布 |
| OpenPI源码、预训练权重、tokenizer和数据 | 选择OpenPI案例时自行配置 | 可替换路径，不依赖原实验目录 |

`launch.py`只接受本目录的vendor；缺失时直接报告错误。prepare把本轮源码复制到新run的runtime，worker不会在训练中回读这个项目的可变源码。冻结runtime的Python包与`jade_lotus`并列，这也是该run自己的源码，不是对原工程的回退。

Python模块命令从本项目的父目录执行，克隆时使用`fragment_training_kit`作为目录名。launcher本身可以用绝对路径从任何工作目录调用。当前按源码目录运行，不需要安装最初实验项目，也没有必须执行的pip editable安装步骤。

## 不申请卡的独立检查

使用已有CPU PyTorch/NumPy/pytest环境；如需安装，先建立自己的环境，再参考`requirements-cpu.txt`。测试会设置CPU检查模式，避免Torch自动发现NPU后端。

从项目目录执行：

```bash
python tools/verify_package.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q tests
python launch.py --help
```

从项目父目录执行CPU数值参考与保存证据复核：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m fragment_training_kit.examples.audit_toy --cpu-oracle --steps 20 --output /tmp/your-unique-toy-oracle.json
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m fragment_training_kit.examples.audit_toy --run-dir fragment_training_kit/validation/normal/probe --archived-evidence --output /tmp/your-unique-normal-audit.json
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m fragment_training_kit.examples.audit_toy --run-dir fragment_training_kit/validation/fault/probe --archived-evidence --output /tmp/your-unique-fault-audit.json
```

输出文件名应使用新名字，已有文件不会被覆盖。归档模式只读取给定副本内部的配置、每步证据和checkpoint，核对原配置SHA、payload、最终模型与Adam；不会访问记录中的原运行路径。它不会启动NPU、HCCL或平台作业。真实新训练仍按README中的prepare → verify → run流程执行，训练与归档审计不能混为一谈。

## 历史记录和支持范围

历史JSON中的原始工作目录、作业ID、配置路径和SHA保留原值，用于追溯2026-11-06真实NPU短跑。它们不是新机器的站点配置。`evidence/source_index.json`中的路径也是历史索引；本项目包含轻量摘要，而不是OpenPI大权重和完整实验输出。

vendored源码保留了主流程所引用的历史模块，以保持已验证源码快照。旧GPU/JAX GradientEngine、里程碑自动评测和LIBERO辅助CLI不是本项目的独立支持入口；其中有旧机器默认路径或未随附脚本。使用者应从`launch.py`和文档中的resident NPU路径接入，不能把vendor里的每个历史CLI都当作通用功能。toy路径不会构造旧GPU/JAX引擎。OpenPI的LIBERO成绩是训练架构案例的历史证据，当前项目不打包LIBERO评测环境。

QZ后端当前限定两个已验证空间，修改站点/空间需移植后端并验收。已有cookie登录不依赖浏览器点击；若改用底层官方加密OAuth凭据流程，还需要Node.js，具体以`vendor/jade_lotus/platform.py`为准。不要把密码或token提交到Git。

## 维护源码与发布

普通使用者无需运行快照维护工具。维护者明确升级运行时时，`tools/snapshot_runtime.py --source-root /explicit/source-tree --refresh`要求来源中同时存在`jade_lotus/*.py`和resident controller，先验证并暂存完成，再替换当前快照；无效来源不会删除现有vendor。来源可以是任意明确选择的源码树，不预设原实验目录。

变更后先运行相关测试及所需真实验收，再执行`python tools/seal_package.py`和`python tools/verify_package.py`。`runtime_manifest.json`绑定vendor源码，`package_manifest.json`绑定当前可分发内容；历史盲测manifest与真实NPU证据保留原版本，不重写成新实验。`.gitignore`隔离本地缓存、凭据、站点设置及新实验输出，但保留随附的轻量JSON和toy state.bin。

本次迁移保留训练controller、模型adapter数学和vendor源码，修正入口/维护工具的原目录假设，并补充可迁移的历史证据审计。迁移本身不等于又完成了一轮NPU训练；真实两卡与受控掉卡结果见06。

2026-11-07迁移验证通过：临时独立副本中41项测试通过；移除外部PYTHONPATH并禁止读取原实验目录后，CPU梯度参考、正常/掉卡两轮归档审计及无提交的prepare/verify均通过。冻结run的全部相关Python导入也来自它自己的runtime。45份vendor源码保持原SHA。检查结果保存在 [迁移验证记录](../validation/standalone-verification-20261107.json)。本轮新增平台作业为0；只使用任务专属暂停文件暂停保活计算，未停止监控父进程，检查后移除自己的文件。

同日的拓扑功能分支另外完成 [独立副本复核](../evidence/topology-standalone-20261107.json)：50 份当前 vendor 源码哈希核验、253 项测试、20 步 CPU 数值参考，以及原正常/掉卡历史证据审计全部通过。子进程通过访问钩子拒绝读取原实验目录及原项目源码目录，外部 PYTHONPATH 被移除。新增真实多卡/切换实验另列在 [09](09_topology_training.md)，不重写原迁移或盲测证据的哈希。当前分支增加并修改了运行时代码，因此不能继续声称这 50 份源码都与最初 45 份快照相同。
