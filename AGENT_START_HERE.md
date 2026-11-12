# 给第一次接手的 agent

本目录提供当前同步昇腾低优策略、通用模型接入点与无需下载数据的回归样例。先读 [当前策略使用手册](docs/12_current_ascend_strategy.md)：空闲节点规划 → LOW/1 单/多卡申请 → 每卡 NPU Adam → 分阶段掉卡恢复 → 累计 16 张完整同步补入。当前代码以默认分支 `main` 为使用入口；`feature/batched-refill` 保留开发历史。历史 OpenPI 64 卡、global256、无重算的 20,000 步/LIBERO 结果，与新补卡路径的 150 步验收是不同证据。

阅读与操作顺序如下。

1. 读 [README](README.md)，确认当前可执行命令及站点配置。
   本目录是独立项目；先读 [独立项目与依赖](docs/07_standalone_project.md)，不要从原实验目录补文件或设置其PYTHONPATH。
2. 读 [架构](docs/01_architecture.md)，弄清原始梯度、归约 buffer、APPLY 与 COMMIT 的区别。
3. 读 [模型接入](docs/02_adapting_your_model.md)，跑 toy 的 CPU 数值参考，再替换模型 adapter。
4. 读 [平台接入](docs/03_platform_and_resources.md)，确认 inventory 的镜像、项目、1/2/4/8/16 卡真实规格、LOW 优先级、空间/资源组与共享预算。
5. 依次执行 prepare → verify → run → audit → observe/cleanup，读 [验收方法](docs/04_validation_and_performance.md)。
6. 若使用整机/碎卡混合申请，读 [拓扑优化与实验](docs/09_topology_training.md)。`cards` 是含候选的物理池预算，`active_cards` 是初始训练成员；旧固定候选模式的 `job_layout` 覆盖物理池，新补卡模式见下一项。节点计划不等于已指定节点，候选初载完成也不等于最新模型/Adam已对齐。不要把分层通信基准误当作已接入故障恢复的训练后端。
7. 需要整作业或连续掉卡恢复时，读 [分阶段恢复](docs/10_pod_loss_recovery.md)。持续补卡进一步读 [累计16张后接入](docs/11_batched_refill.md)，显式启用 `resident_pod_refill`；不足16张仅保留资源，完整状态对齐后才入组。该模式下初始 `job_layout` 仅覆盖 `active_cards`，与旧候选槽位/切换互斥。按对应版本的真实证据判断支持范围。

推广接入时优先使用 [完整初始组配置](config/site.low_priority.example.json) 和手册中的 `topology.py prepare` 流程。若先起 48 张、目标 64 张，使用 [部分初始组配置](config/site.refill.example.json) 和 `launch.py prepare`，或按手册单独规划 48 张；当前 `topology.py prepare` 不直接支持这种 48/64 差额。不要把 `topology.py watch --fill`、旧 `candidate_watch.py` 当作新动态补卡的驱动，也不要混用旧槽位/切换参数。

首轮建议两张卡、20 次更新。使用新的绝对路径作为 run directory；先完成无注入的正常短跑，再单独做有界受控失联实验。成功标记、最终完整状态一致、数值参考和平台官方终态缺一不可。不要把终端没有报错当成验收。

通常只需要向使用者确认一次站点信息：NPU 环境 Python、CANN set_env.sh、共享目录、资源 inventory、私有登录文件和全局预算。登录密码不进入配置、日志或本工具包；使用已有私有 cookie 文件或平台支持的 OAuth。得到站点配置与资源授权后直接执行，不重复请求同一授权。

需要适配自己的模型时，优先改 `examples/model_adapter_template.py` 的副本和自己的训练配置。初次不要改 vendor 中的成员协议或恢复流程；这些变化需要新的故障与数值验收。

遇到问题时记录：命令、退出码、run 路径、最早原始错误、是否已创建平台作业、当前自有资源数。保留失败运行，新建版本重跑，不能覆盖失败证据。清理只作用于自己 run 的作业。

参考证据与已知边界见 [真实案例](docs/05_evidence_and_limits.md)。本工具包的短跑结果见 [接入验收](docs/06_handoff_acceptance.md)。
