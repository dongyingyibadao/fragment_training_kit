# 本次真实接入的证据

`blind-*` 由不继承主会话的agent在独立工具包副本中产出。`final-normal-audit.json`、`final-fault-audit.json`是主会话使用增强覆盖/计时检查，对相同原始训练证据做的只读复核。

`normal/`、`fault/`保存配置、完整提交/阶段/最终状态证据、小模型checkpoint和平台收尾。原始执行目录和配置绝对路径保留，用于溯源；副本不冒充在另一台机器已经重新执行的训练。现在可用 `--run-dir 本地副本/probe --archived-evidence` 仅读取本地配置/状态并重新核验，不回读原始绝对路径。实际重新训练仍应先prepare自己的新run，再对其probe执行audit；两种方式的区别与命令见 [独立项目说明](../docs/07_standalone_project.md)。

`blind-input-manifest.json`标记首次交给冷启动agent的目录版本。之后发布前补齐通用模板的合同字段、提前配置预检、审计的覆盖/故障计时检查和文档措辞；实际NPU训练控制流、adapter数学和toy模型均未修改。原始盲测审计与增强审计都保留。

这里只含本次测试的证据与tiny模型状态，没有站点登录文件、inventory或大模型权重。平台官方证明记录两轮各自的两条LOW/1作业，全部job_succeeded，running/pending/reserved均为0。两轮卡均位于同一物理节点，本次短跑不替代跨节点大模型网络性能测试。

`standalone-verification-20261107.json`记录迁为同级独立项目后的验证：临时独立副本、外部PYTHONPATH移除、原实验目录禁读、41项测试、CPU数值参考、两轮归档证据审计、prepare-only和冻结runtime导入。它不创建平台作业，不是第二次NPU短跑。`blind-source-fingerprint.json`与两份`blind-*-detailed-metrics.json`是原验收的小型附加证据，内容未重写。
