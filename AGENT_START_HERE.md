# 给第一次接手的 agent

本目录整理了已在 OpenPI 上完成 64 张独立 LOW/1 昇腾卡、global batch 256、关闭梯度重算、20,000 步训练的架构。它提供一个通用模型接入点和一个无需下载数据的回归样例。阅读与操作顺序如下。

1. 读 [README](README.md)，确认当前可执行命令及站点配置。
   本目录是独立项目；先读 [独立项目与依赖](docs/07_standalone_project.md)，不要从原实验目录补文件或设置其PYTHONPATH。
2. 读 [架构](docs/01_architecture.md)，弄清原始梯度、归约 buffer、APPLY 与 COMMIT 的区别。
3. 读 [模型接入](docs/02_adapting_your_model.md)，跑 toy 的 CPU 数值参考，再替换模型 adapter。
4. 读 [平台接入](docs/03_platform_and_resources.md)，确认 inventory 的镜像、项目、单卡规格、LOW 优先级和共享预算。
5. 依次执行 prepare → verify → run → audit → observe/cleanup，读 [验收方法](docs/04_validation_and_performance.md)。

首轮建议两张卡、20 次更新。使用新的绝对路径作为 run directory；先完成无注入的正常短跑，再单独做有界受控失联实验。成功标记、最终完整状态一致、数值参考和平台官方终态缺一不可。不要把终端没有报错当成验收。

通常只需要向使用者确认一次站点信息：NPU 环境 Python、CANN set_env.sh、共享目录、资源 inventory、私有登录文件和全局预算。登录密码不进入配置、日志或本工具包；使用已有私有 cookie 文件或平台支持的 OAuth。得到站点配置与资源授权后直接执行，不重复请求同一授权。

需要适配自己的模型时，优先改 `examples/model_adapter_template.py` 的副本和自己的训练配置。初次不要改 vendor 中的成员协议或恢复流程；这些变化需要新的故障与数值验收。

遇到问题时记录：命令、退出码、run 路径、最早原始错误、是否已创建平台作业、当前自有资源数。保留失败运行，新建版本重跑，不能覆盖失败证据。清理只作用于自己 run 的作业。

参考证据与已知边界见 [真实案例](docs/05_evidence_and_limits.md)。本工具包的短跑结果见 [接入验收](docs/06_handoff_acceptance.md)。
