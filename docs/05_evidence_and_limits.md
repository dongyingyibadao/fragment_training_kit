# OpenPI案例、实测结论与适用范围

OpenPI是该架构的训练对象案例。当前已验证的核心路径是：每卡完整模型/Adam常驻NPU，原生梯度SUM，APPLY前保护原local_grad，按版本/会话固定成员，失联后缩组补缺失单元，替补完整状态对齐后入组。

| 实验 | 实际结论 | 母项目中的证据 |
|---|---|---|
| 32个LOW/1、global256、无重算、500步 | 受控单卡退出后31健康PID保留，只补8样本；设备入组后恢复32卡；最终模型/Adam hash一致 | `artifacts/ascend-resident-20261103/native-fastpoll32-rdma16-devicejoin-500-independent-audit-v1.json` |
| 未预告平台单卡退出 | 一次APPLY前失联检测与8→7缩组更新通过 | `artifacts/ascend-resident-20261103/unknown-platform8-rdma16-independent-audit-v1.json` |
| 64卡全参数OpenPI长跑 | 绝对step20000完成；最后段由16000恢复并新增4000步；64卡最终完整状态一致 | `runs/pi05-ascend-resident-low64-b4-rdma12-resume16000-20000-20261105-v1/probe/summary.json` |
| 优化后的64卡新增2000步 | 正常更新均值2.732秒、P95 2.952；不能当任意模型速度保证 | `docs/ascend_collective_slowdown_results_20261105_zh.md`及其中独立审计 |
| 最终模型LIBERO | 40任务×50初态，1899/2000成功，94.95%；Spatial97.4%、Object96%、Goal96%、LIBERO10 90.4% | `artifacts/libero-eval-20000-20261106/final-report/audit.json` |

32卡500步恢复32卡后的447步均值为2.572秒，固定原生DDP短跑完成间隔约1.661秒。两轮放置/时段等不同，不能当严格受控训练质量/吞吐等价结论。最终LIBERO也没有同期、同评测噪声协议的原生集群训练baseline。

本目录 `evidence/` 保存这些结论的轻量摘要及来源SHA，完整大权重、数据和海量逐步日志不包含在分发包内。`vendor/` 是提供通用路径所需运行时源码快照；最初OpenPI长跑冻结代码仍保留在原run中，新增adapter是显式扩展，要由本次短实验单独验收。

2026-11-07 的新拓扑分支另有 4×16 OpenPI 30 步短训（提交间隔均值 1.767 s）、600 步小模型自动切入/定时切回的严格数值验收，以及打包/单卡与 timeout 往返通信对照。见 [09 的完整记录](09_topology_training.md)。这些是新增证据，不替代上述长跑、单卡故障或 LIBERO 成绩，也不能把小模型切换成功扩大成 OpenPI 整 pod 突然失联已能自动恢复。

## 清楚区分已支持与需要继续实现的部分

- 初始global256、atomic4、原生AdamW、2–64卡范围是当前接口合同；不是任意batch/优化器框架。
- 支持一次APPLY前失联的健康状态保留、缩组重算缺失分配；设备恢复需要同版本环境和实际故障验证。
- 加载中的替补不参与当前步；完整状态同步入组有边界暂停，不是零等待。
- 通用probe没有集成任意Adam写入中点的自动恢复。单独apply-loss实验覆盖的狭窄时点不能扩大成生产路径已全部支持。
- 修复期间再次失联、任意多卡同时失联、CPU controller崩溃后的自动续跑、独立持久梯度接受、任意新worker自动弹性扩缩均未作为本工具包保证。
- 原生BF16 moments没有FP32 master；与其他精度/优化器的轨迹等价要独立验证。
- dropout、动态参数、分片优化器、可变前向buffer等需要额外合同，不能由toy通过推断它们都可直接接入。
- “自动占卡”仅对配置与自有账本中的任务执行；用户私有登录、数据权限、网络与共享存储仍由站点提供。

这些界限是接入时的工程条件。遇到超出范围的故障应保留证据并从完整checkpoint恢复，不能为了继续跑而跳过版本、样本覆盖或状态一致性检查。
