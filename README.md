# 低代码强化学习教学平台

面向教学场景的强化学习实验原型平台。项目提供可视化实验配置、流式训练、教师基准对标、教学作业模板、课堂分析与 Markdown 报告导出，适合用于离散环境强化学习课程演示与毕业设计原型展示。

## 项目状态

- 当前仓库是原型系统，不是生产环境应用。
- 已支持教师/学生视角下的不同操作入口。
- 教师基准的增删改现在只允许教师视角触发。
- 角色仍然是原型级元数据，不是完整的登录鉴权系统。

## 功能概览

- 支持 `GridWorld`、`CliffWalking`、`WindyGridWorld`、`FrozenLake` 四类离散环境。
- 支持 `Q-Learning`、`SARSA`、`DQN`、`REINFORCE` 四种强化学习算法。
- 支持表单化环境配置、训练参数配置与实验命名。
- 支持基于 WebSocket 的流式训练、暂停、继续与取消。
- 支持奖励、成功率、`epsilon`、TD 误差等训练指标可视化。
- 支持教学作业模板的一键加载与实验参数回填。
- 支持内置教师基准与教师自定义基准。
- 支持当前实验结果与教师基准的逐项对标评估。
- 支持课堂分析面板、作业维度统计与基准通过率统计。
- 支持实验历史记录、轨迹回放、多次运行对比与策略网格展示。
- 支持将实验结果导出为 Markdown 教学报告。

## 技术栈

- 后端：`FastAPI`、`Pydantic v2`、`NumPy`、`PyTorch`
- 前端：`Vue 3`、`TypeScript`、`Vite`
- 存储：`SQLite`

## 仓库结构

```text
.
├─ backend/                 FastAPI 后端与训练逻辑
│  ├─ app/
│  │  ├─ api/               HTTP / WebSocket 路由
│  │  ├─ repositories/      SQLite 持久化
│  │  ├─ rl/                环境与算法实现
│  │  ├─ schemas/           Pydantic 数据结构
│  │  └─ services/          作业模板、教师基准、分析与报告服务
│  └─ tests/                后端测试
├─ frontend/                Vue 单页应用
│  ├─ public/
│  └─ src/
├─ docs/                    项目文档
└─ data/                    运行期输出目录（默认 SQLite 数据库位置）
```

## 核心概念

### 作业模板

作业模板是面向学生的教学任务预设，包含：

- 作业标题、简介、说明与学习目标
- 对应的环境、算法与训练参数
- 可选的默认教师基准绑定

应用作业模板后，系统会直接回填实验配置，适合作为课堂练习或课程作业的起点。

### 教师基准

教师基准是面向结果评估的参考标准，包含：

- 一组教师侧实验配置
- 平均奖励、最佳奖励、成功率、稳定窗口成功率等阈值
- 教师备注与课堂解释说明

学生可以查看并选择教师基准进行对标；教师可以在教师视角下保存、更新和删除自定义基准。

## 快速开始

### 1. 启动后端

```powershell
cd backend
python -m pip install -e .
uvicorn app.main:app --reload
```

默认地址：

- API：`http://127.0.0.1:8000`

### 2. 启动前端

```powershell
cd frontend
Copy-Item .env.example .env
npm install
npm run dev
```

默认地址：

- 前端：`http://127.0.0.1:5173`

前端默认通过 `frontend/.env` 中的 `VITE_API_BASE_URL` 连接后端；仓库提供了 `frontend/.env.example` 作为示例。

## 常用接口

- `GET /api/catalog`：获取环境与算法参数定义
- `GET /api/assignments`：获取内置教学作业模板
- `GET /api/benchmarks`：获取内置与自定义教师基准
- `POST /api/benchmarks`：创建自定义教师基准
- `PUT /api/benchmarks/{benchmark_id}`：更新自定义教师基准
- `DELETE /api/benchmarks/{benchmark_id}`：删除自定义教师基准
- `GET /api/analytics/classroom`：获取课堂分析数据
- `GET /api/experiments/history`：获取实验历史
- `GET /api/experiments/{run_id}`：获取单次实验详情
- `POST /api/experiments/run`：同步执行实验
- `WS /api/experiments/stream`：流式执行实验
- `POST /api/reports/render`：导出 Markdown 报告

## 数据与输出

- 运行结果默认保存在仓库根目录的 `data/experiments.db`。
- 教师自定义基准也会保存到同一个 SQLite 数据库中。
- `backend/data/` 和 `backend/tmp_eval_*` 目录主要是开发/测试期遗留数据，不是运行时主数据目录。

## 开发与验证

后端语法检查：

```powershell
python -m compileall backend/app
```

前端构建：

```powershell
cd frontend
npm run build
```

后端测试：

```powershell
cd backend
python -m pytest
```

## 当前限制

- 当前“教师/学生”角色仍然依赖前端视角与请求头传递，不等同于完整身份认证。
- 后端 CORS 目前为原型阶段的宽松配置。
- 自定义教师基准的权限控制已做接口层限制，但仍建议后续接入真实登录系统。

## 公开仓库前的建议

- 不要提交本地 `.env`、数据库文件、缓存目录、打包产物和 IDE 配置。
- 运行数据应保留在本地或单独发布为示例数据，而不是与源码混放提交。
- 若仓库历史中已经跟踪过 `data/`、`.idea/`、`__pycache__/`、`frontend/dist/` 等内容，公开前应先从 Git 索引中移除。
- 若前端图片、图标或其他素材来自第三方，请在公开前补充来源与授权说明。

## 文档

- [系统架构说明](docs/architecture.md)

## 后续迭代方向

- 接入真实身份认证与权限系统
- 支持教师自定义作业模板而不仅限于内置模板
- 增强课堂评分、导出和多作业统计能力
- 强化轨迹回放与基准对比的教学解释能力
