# 网页自动占卡与平台配置

当前经过验证的后端是 QZ（`qz.sii.edu.cn`）。所谓“网页自动占卡”使用的是官方网页调用的后台接口：发现项目、镜像和资源规格，提交 LOW/1 作业，跟踪平台状态与自有提交账本。默认独立单卡，也支持已发现的 2/4/8/16 卡规格；节点资源规划和混合布局见 [09](09_topology_training.md)。运行不依赖模拟鼠标点击；训练进程也不保存网页登录密码。

身份来自使用者已有的私有 SSO cookie 文件。底层平台模块也提供官方 OAuth client；具体 launcher 已开放的参数见 `launch.py --help`。cookie 权限应为0600，路径在命令行提供；分发目录和源码快照不包含 cookie、密码、token。已有 SSO 可沿官方 HTTPS 登录链刷新；登录过期时按官方入口重新登录，不把凭据写进训练日志。

## 站点配置

`config/` 的示例描述以下内容，正式字段以示例和 `launch.py` 为准：

- 控制端和作业环境都能调用的 Python 绝对路径、CANN `set_env.sh`、双方都可访问的共享 run directory。当前prepare也会检查控制端可见该CANN脚本；若脚本只存在镜像内，需要移植该预检方式，而不是填一个不存在的本机路径。
- discovery 输出的 inventory；选定项目、镜像、单卡昇腾规格、LOW level 与其可用资源。
- 初始卡数、全局共享预算、最大作业时长、训练步数、adapter配置。
- HCCL/CANN环境、通信桶、控制轮询、心跳/阶段超时、入组传输方式。
- 全局 registry 路径：同一账号同时运行的这些工具实例使用同一个文件，避免各自计算预算。

本 QZ 后端的授权空间白名单当前只覆盖已验证的两个空间；换到其他空间或其他平台，需要显式扩展后端的准入配置并验证平台 API、镜像与优先级语义。不能仅替换 URL 就推断其他平台也可用。QZ里`task_priority=1`与`LOW`的对应关系来自已核验的平台定义，不能类推到别的平台。

## 提交前后有什么区别

prepare只创建本地新运行目录、冻结代码与配置；verify检查冻结文件。run是显式开始执行：启动CPU controller并按冻结的布局提交预算内的作业。observe查询，不创建新作业。cleanup仅处理本run的自有job，直到官方终态才释放账本预留。未知提交结果仍占预算，不能在超时后盲目重复POST。

`ascend_fleet/state.json`保存自有job_id、不可变请求和观察结果；`latest.json`显示预留、running、pending和错误。平台API不可用时，不以本地进程退出代替官方释放证明。要区分 requested_reserved_cards、真实执行worker证据和平台可返回的allocated identity；某字段不可核验不代表没有用卡。

初始训练成员的ID在本轮固定。新提交的任意worker不会自动成为READY，更不能在旧版本上参与下一步。当前launcher按固定初始成员申请，之后以观察和已支持的supervisor替补流程为主，不无限补占不被controller接纳的作业。跨所有已登记run的预算上限最多64卡，本工具包首轮默认2卡。

底层controller有显式checkpoint/source-spec恢复接口，但新launcher尚未暴露一键resume。需要整组新run恢复时，接入者应根据底层恢复合同补充配置和验证；不能把一个旧checkpoint路径随意放进site就假定续训已接好。

## 设备与环境

CANN set_env之后保留其PYTHONPATH和库路径，再追加本工具包运行时代码。平台可能将物理第12张卡映射为容器里的逻辑npu0；不要强制ASCEND_VISIBLE_DEVICES=0覆盖已有mask。单卡worker核验一个可见NPU并使用逻辑npu0；多卡作业保留完整可见集合，各worker按独立逻辑序号选设备，并核对作业总可见数量。

现有实测参考设置为 HCCL_INTRA_ROCE_ENABLE=1、RDMA_TIMEOUT=12、较大通信buffer与512MiB桶、10ms热控制轮询。它们对当前910B/CANN/网络布局有效，不能当通用最优值。RDMA_TIMEOUT是底层事务参数，不等于controller的心跳或作业超时；不要为了提速把所有超时都缩短。具体调参方法见 [性能验证](04_validation_and_performance.md)。

当前恢复使用torch_npu的设备停止/重启与HCCL取消接口。Python、torch、torch_npu、CANN和驱动版本需要成套可用。不要升级共享环境来试运行此样例；优先使用已验证环境或独立镜像。
