**独立目录首次接入验收：通过**

以下是2026-11-06原始运行的验收说明。历史运行目录仅用于溯源；可查看的轻量证据已随本项目保存，链接指向本地副本，不需要访问原项目。

独立副本无额外站点交互完成CPU数值验收、真实LOW/1两卡20步正常训练及受控失联2→1的20步训练；两轮数值、实际样本覆盖与最终checkpoint均通过，自有4项平台作业全部官方成功终态且资源为0。

仅按独立副本文档和现成site执行，没有补问站点信息或权限。外部PYTHONPATH已移除，实际jade_lotus来自kit/vendor；两个冻结run也分别从自身runtime导入，工具包和冻结源码均未修改。

独立目录：`/inspire/sj-ssd3/project/robot-decision/laijunxi-CZXS25230141/Steadfast_as_the_Jade_Lotus/artifacts/fragment-training-kit-20261106/isolated/fragment_training_kit`。源码整体SHA256：`b70b738159c3e4a0eff489cf4dc8248d674fa9213b1706505a9019a9e3f3a95c`；runtime manifest SHA256：`439a4ddda2da068a1122ef159ee28f4133af51e6ae82a4073be1e510cda94fa3`。逐文件指纹见[来源指纹](blind-source-fingerprint.json)。

已读README、AGENT_START_HERE、PLAN及docs/01至06；按package verify → CPU oracle → prepare → frozen verify → run → audit → observe → fresh official query执行。先完成正常轮官方释放，再开始受控故障轮。两轮全部命令退出码为0；[命令journal](blind-command-journal.jsonl)保存精确argv、工作目录、环境、时间和原始日志路径。

| 真实实验 | 步数 | loss | 参数最大误差 | Adam最大误差 | 正常更新均值 | 实际commit间隔均值 | 启动至释放 |
|---|---:|---|---:|---:|---:|---:|---:|
| 正常两卡 | 20 | 4.453125 → 0.167660490 | 8.94e-08 | 1.19e-07 | 0.094098s | 0.105509s | 98.246s |
| 受控失联2→1 | 20 | 4.453125 → 0.167660519 | 2.98e-08 | 2.98e-07 | 0.094752s | 0.105875s | 96.913s |

速度均排除前两步预热及故障步；实际commit间隔包含周期全状态检查暂停。正常轮18步、故障轮17步参与均值。每worker的数据、反传、通信、更新计时及检查暂停另存详细指标，未相加不同rank的阶段最大值。两轮首步分别为1.647156s、1.666493s。

受控故障发生于第3步：w000002完成本地反传并在collective前被退出，健康w000001原PID/session保留。恢复前后模型、Adam、梯度和存储指针相同，只补算丢失的32个奇数atomic单元=128样本；回滚0、最终成员1。该更新耗时1.162509s，实际commit间隔1.167542s，worker恢复计时0.572367s（与部分阶段重叠，不能相加）。

CPU oracle在2/3/7分片及clear=False补算下最大逐参数梯度误差为0。真实训练每步原始+补算单元唯一覆盖0–63，与commit分配逐项一致，每步256样本、重算0；NPU对CPU使用atol=rtol=2e-4，两轮最终完整checkpoint均核验通过。正常轮两卡最终完整状态相同；故障轮唯一健康成员完成20步。

[CPU验收](blind-cpu-oracle.json)、[正常数值审计](blind-real-npu-audit.json)、[故障数值审计](blind-fault-real-npu-audit.json)；[正常详细指标](blind-normal-detailed-metrics.json)、[故障详细指标](blind-fault-detailed-metrics.json)。

资源证据对各自账本中的job逐项重新调用官方只读detail/instances接口并匹配身份；共4个LOW/1作业全部job_succeeded。两轮active/running/pending/requested_reserved均为0，原controller精确身份均已退出；最高同时自有预留2卡。未停止其他作业或修改共享环境。证据：[正常资源终态](blind-normal-resource-proof.json)、[故障资源终态](blind-fault-resource-proof.json)，各run同时保留cleanup_receipt.json。

接入无阻断、无额外人工交互。需要留意：COMPLETE后等待官方终态传播会触发自有cleanup并留下STOP；受控设备恢复的FORCE STOP/107022是本轮预期错误；原副本audit的故障轮steady均值包含恢复步，本记录已更正口径并保留原始结果。checkpoint独立写入耗时没有直接仪表数据，已仅记录最终化间隔和verify耗时。

本次结果只覆盖小型FP32 toy与一次APPLY前子进程失联，supervisor存活且无替补；不代表整平台作业突失、多次失联或大模型吞吐。完整机器记录：[blind-acceptance.json](blind-acceptance.json)。
