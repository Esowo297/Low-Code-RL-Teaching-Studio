# 低代码强化学习教学平台

本仓库包含一个毕业设计原型项目，用于实现面向教学场景的低代码交互式强化学习平台。

## 当前范围

- `backend/`：基于 FastAPI 的后端服务，包含实验数据结构、可配置的 GridWorld、CliffWalking 与 WindyGridWorld 环境，以及可运行的 Q-learning、SARSA、DQN 和 REINFORCE 训练流程。
- `frontend/`：基于 Vue 3 的单页界面，用于实验配置、执行控制、算法切换和结果可视化。
- `docs/`：可直接复用于论文写作的系统架构说明。
- `data/`：运行期输出目录，用于保存实验结果。

## 当前实现功能

- 面向 GridWorld、CliffWalking 与 WindyGridWorld 教学场景的表单化实验设计
- 支持参数化配置的 Q-learning、SARSA、DQN 和 REINFORCE
- 基于 WebSocket 的训练指标实时推送
- 支持流式训练的暂停、继续和取消控制
- 输出奖励值、epsilon 和 TD 误差等训练指标
- 教师基准预设及一键参数加载
- 内置可复用任务元数据的教学作业模板
- 支持按环境维度区分的教师基准、作业模板和课堂分析
- 基于教师阈值的基准通过/未通过评估
- 已完成实验的 Markdown 教学报告导出
- 含教师/学生角色信息的提交元数据与课堂记录
- 面向教师的分析面板，支持班级实验汇总、作业进度和基准进度查看
- 针对当前实验及已保存实验的逐步轨迹回放
- 已保存实验的多次运行对比面板
- 基于 SQLite 的实验持久化，并兼容旧版 JSON 导入
- 训练后策略网格渲染
- 支持矩形网格、悬崖单元、风力列提示和轨迹回放的环境可视化
- 最近实验的历史记录保存

## 启动后端

```powershell
cd backend
python -m pip install -e .
uvicorn app.main:app --reload
```

API 默认启动地址为 `http://127.0.0.1:8000`。

可用实验接口如下：

- `GET /api/assignments`：内置教学作业模板
- `GET /api/benchmarks`：教师定义的基准预设
- `GET /api/analytics/classroom`：班级级别实验聚合与基准进度
- `POST /api/experiments/run`：同步执行
- `WS /api/experiments/stream`：实时流式执行
- `POST /api/reports/render`：生成与基准对齐的 Markdown 实验报告

## 启动前端

```powershell
cd frontend
npm install
npm run dev
```

界面默认启动地址为 `http://127.0.0.1:5173`。

## 后续建议迭代方向

- 在当前角色元数据基础上补充认证与真实访问控制能力
- 支持教师自行管理作业与模板，而不仅限于内置预设
- 增强轨迹回放标注和基准回放对比能力
- 增加评分导出和多作业班级分析能力
