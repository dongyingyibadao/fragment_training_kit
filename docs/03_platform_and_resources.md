# 网页自动占卡与平台配置

当前经过验证的后端是 QZ（`qz.sii.edu.cn`）。所谓“网页自动占卡”使用的是官方网页调用的后台接口：发现项目、镜像和资源规格，提交 LOW/1 作业，跟踪平台状态与自有提交账本。默认独立单卡，也支持已发现的 2/4/8/16 卡规格；节点资源规划和混合布局见 [09](09_topology_training.md)。运行不依赖模拟鼠标点击；训练进程也不保存网页登录密码。

当前组合的完整命令见 [策略使用手册](12_current_ascend_strategy.md)：启动前用节点视图规划初始布局；训练中开启恢复与 refill，由 launcher 在缺口/预算内再次查询节点并申请候选，累计 16 张完整同步补入。初始完整组推荐 [site.low_priority.example.json](../config/site.low_priority.example.json)；48 初始/64 目标使用 [site.refill.example.json](../config/site.refill.example.json)，两者的 prepare 入口不同，按手册执行。

身份来自使用者已有的私有 SSO cookie 文件。底层平台模块也提供官方 OAuth client；具体 launcher 已开放的参数见 `launch.py --help`。cookie 权限应为0600，路径在命令行提供；分发目录和源码快照不包含 cookie、密码、token。已有 SSO 可沿官方 HTTPS 登录链刷新；登录过期时按官方入口重新登录，不把凭据写进训练日志。

## 站点配置

`config/` 的示例描述以下内容，正式字段以示例和 `launch.py` 为准：

- 控制端和作业环境都能调用的 Python 绝对路径、CANN `set_env.sh`、双方都可访问的共享 run directory。当前prepare也会检查控制端可见该CANN脚本；若脚本只存在镜像内，需要移植该预检方式，而不是填一个不存在的本机路径。
- discovery 输出的 inventory；选定项目、镜像、单卡昇腾规格、LOW level 与其可用资源。
- 初始卡数、全局共享预算、最大作业时长、训练步数、adapter配置。
- HCCL/CANN环境、通信桶、控制轮询、心跳/阶段超时、入组传输方式。
- 全局 registry 路径：同一账号同时运行的这些工具实例使用同一个文件，避免各自计算预算。

本 QZ 后端的授权空间白名单当前只覆盖已验证的两个空间；换到其他空间或其他平台，需要显式扩展后端的准入配置并验证平台 API、镜像与优先级语义。不能仅替换 URL 就推断其他平台也可用。QZ里`task_priority=1`与`LOW`的对应关系来自已核验的平台定义，不能类推到别的平台。

## 从原始发现生成可用 inventory

按当前手册先运行不带 selectors 的只读 discover，保存 `raw-discovery.json`。查看其中 `workspaces` 的项目、成功镜像与调度定义，以及 `groups` 中的资源规格。默认项目名或多个镜像可能无法唯一匹配；原始发现也不会自行核实 LOW。不要直接将这个结果交给 prepare。

在项目之外创建自己的 `/shared/private/workspace-selectors.json`。以下仅展示一项结构；将项目/镜像替换成原始结果里的精确 ID，并按授权空间分别配置。`priority` 只有在核实平台定义后才能填写为已验证；需要限定资源组时增加 `group_ids` 数组。

```json
[
  {
    "workspace_id": "ws-e5a87523-5ec2-4afa-a14a-56c3f4088d14",
    "project_id": "REPLACE_WITH_DISCOVERED_PROJECT_ID",
    "image_id": "REPLACE_WITH_DISCOVERED_SUCCESS_IMAGE_ID",
    "priority": {
      "verified": true,
      "level": "LOW",
      "name": "1",
      "task_priority": 1
    }
  }
]
```

从项目父目录重新发现，显式传入该文件：

```bash
python -c 'import fragment_training_kit; from jade_lotus.ascend_platform import main; raise SystemExit(main())' discover --workspace-config /shared/private/workspace-selectors.json --cookie-file /shared/private/qz.cookies --output /shared/private/verified-ascend-inventory.json
```

检查命令摘要和 inventory 的 `errors`、`selections`；只有项目与镜像唯一匹配且所需合法规格都存在才继续。未配置空间仍可能产生原始记录；仅保留已明确授权、核实的 selections，并使 site 的 `workspace_ids`/`allowed_group_ids` 与之匹配。多种规格用于后续动态补卡，不能只保留本次初始 16 卡规格。`verified=true` 是使用者基于平台定义的确认声明，命令不会代替这项核实。

## 提交前后有什么区别

prepare只创建本地新运行目录、冻结代码与配置；verify检查冻结文件。run是显式开始执行：启动CPU controller并按冻结的布局提交预算内的作业。observe查询，不创建新作业。cleanup仅处理本run的自有job，直到官方终态才释放账本预留。未知提交结果仍占预算，不能在超时后盲目重复POST。

`ascend_fleet/state.json`保存自有job_id、不可变请求和观察结果；`latest.json`显示预留、running、pending和错误。平台API不可用时，不以本地进程退出代替官方释放证明。要区分 requested_reserved_cards、真实执行worker证据和平台可返回的allocated identity；某字段不可核验不代表没有用卡。

初始布局和成员映射在本轮冻结。普通模式不会持续申请任意新 worker；显式 refill 模式通过新意图和成员映射追加候选历史，不修改初始布局，也不复用旧身份。申请成功不等于 READY，READY 不等于已正式准入，更不能在旧模型/Adam 版本上参与下一步。累计少于 16 张只保留资源，完整批次加载/同步/准入均由 controller 授权。跨所有已登记 run 的共享上限最多 64 卡，包括活动、候选、排队、未知与未确认停止；首轮接入样例仍为两卡。

节点计划核对 Ready、NPU/CPU/内存并倾向较大合法作业，没有空闲容量时才排队单卡；当前不能指定物理节点，也不会因出现更完整的空闲节点而自动迁走满预算的健康活动组。`topology.py watch --fill` 只补固定初始槽位，动态补卡使用 `launch.py run`。`budget_scope=account` 仅支持只读规划，不能据此执行申请；其他账号项目的作业不由本工具随意停止。

底层controller有显式checkpoint/source-spec恢复接口，但新launcher尚未暴露一键resume。需要整组新run恢复时，接入者应根据底层恢复合同补充配置和验证；不能把一个旧checkpoint路径随意放进site就假定续训已接好。

## 设备与环境

CANN set_env之后保留其PYTHONPATH和库路径，再追加本工具包运行时代码。平台可能将物理第12张卡映射为容器里的逻辑npu0；不要强制ASCEND_VISIBLE_DEVICES=0覆盖已有mask。单卡worker核验一个可见NPU并使用逻辑npu0；多卡作业保留完整可见集合，各worker按独立逻辑序号选设备，并核对作业总可见数量。

现有实测参考设置为 HCCL_INTRA_ROCE_ENABLE=1、RDMA_TIMEOUT=12、较大通信buffer与512MiB桶、10ms热控制轮询。它们对当前910B/CANN/网络布局有效，不能当通用最优值。RDMA_TIMEOUT是底层事务参数，不等于controller的心跳或作业超时；不要为了提速把所有超时都缩短。具体调参方法见 [性能验证](04_validation_and_performance.md)。

当前恢复使用torch_npu的设备停止/重启与HCCL取消接口。Python、torch、torch_npu、CANN和驱动版本需要成套可用。不要升级共享环境来试运行此样例；优先使用已验证环境或独立镜像。
