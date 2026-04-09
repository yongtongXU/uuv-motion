# UUV Motion

一个用于 **6 台同构水下机器人（UUV）** 的跑点仿真与网络发送项目，包含：
- 全覆盖梳状路径规划（支持返航起点）
- 多进程运动仿真并生成轨迹 CSV
- 多进程 TCP 按帧发送路径点
- GUI 一站式控制（配置、规划、仿真、发送、状态监控）

## 1. 坐标系约定
本项目使用 **NED** 坐标系：
- `X`：North（北，正方向）
- `Y`：East（东，正方向）
- `Z`：Down（向地心，正方向）

`config/init.json` 中已写明：`"world_frame": "NED"`。

---

## 2. 目录结构

```text
uuv-motion/
├─ config/
│  ├─ init.json            # 场景与UUV初始配置
│  ├─ network_format.json  # TCP 报文模板与正则转换规则
│  └─ network_endpoints.json # 每个uuv_id对应的IP/端口配置
├─ main.py                  # GUI 主入口（推荐）
├─ mian.py                  # 兼容入口（历史拼写）
├─ planning.py              # 路径规划（输出 path.json）
├─ motion_server.py         # 多进程轨迹仿真（输出 trajectories/*.csv）
├─ network.py               # 多进程 UDP 发送端
├─ show_planned_paths.py    # 规划路径图输出
├─ show_trajectory_cv.py    # OpenCV 轨迹回放
├─ path.json                # 规划结果
├─ trajectories/            # 轨迹输出目录
├─ requirements.txt         # Python 依赖
└─ README.md
```

---

## 3. 环境要求
- Python 3.10+（建议 3.10/3.11）
- Windows/Linux/macOS
- GUI 模式需要 `tkinter`（Python 自带，但安装时需包含 Tk 支持）

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 4. 快速开始

### 4.1 启动 GUI（推荐）

```bash
python main.py
```

GUI 内支持：
- 读取/写入 `init.json`
- 编辑每台 UUV 初始状态
- 设置区域范围（`area.length/area.width`）
- 一键规划
- 一键运行 motion
- 一键发送 / 暂停发送 / 停止发送
- 网络连接检测（TCP）
- 每台 UUV 独立 IP+端口配置
- 实时状态显示（待命、正在发送、暂停发送、发送结束）

### 4.2 命令行模式

仅规划：
```bash
python planning.py
```

运行运动仿真并生成轨迹：
```bash
python motion_server.py
```

TCP 发送（多进程）：
```bash
python network.py
```

GUI 的 CLI 兼容入口：
```bash
python main.py --cli
```

---

## 5. 配置文件说明

## `config/init.json`
关键字段：
- `scene.sample_rate_hz`：系统帧率（例如 `50`）
- `scene.world_frame`：坐标系（NED）
- `area.length`、`area.width`：任务区域
- `planner.type`：规划器类型（当前 `comb_no_obstacle`，预留 `comb_with_obstacle`）
- `uuv_template.sonser`：探测宽度（用于梳状覆盖间距）
- `uuvs[]`：每台 UUV 初始状态（`id` + `pose`）
- `environment.obstacles[]`：障碍物列表（预留给障碍物规划）

### 障碍物参数（预留接口）
后续接入障碍物规划时，建议在 `config/init.json -> environment.obstacles` 使用以下结构：

- 通用字段：
  - `id`：障碍物唯一标识
  - `enabled`：是否启用（`true/false`）
  - `type`：几何类型，建议 `circle` / `rectangle` / `polygon`
  - `z_min`、`z_max`：障碍物生效深度范围（NED 下为正向下，单位 m）
  - `safety_margin_m`：安全膨胀半径（单位 m）
- `circle` 类型：
  - `center`：`{"x": ..., "y": ...}`
  - `radius`：半径（m）
- `rectangle` 类型：
  - `center`：`{"x": ..., "y": ...}`
  - `length`、`width`：长宽（m）
  - `yaw_deg`：绕 Z 轴旋转角（度）
- `polygon` 类型：
  - `points`：顶点数组，按顺/逆时针给出，例如 `[{"x":0,"y":0},{"x":10,"y":0}, ...]`

示例：

```json
"environment": {
  "obstacles": [
    {
      "id": "obs_c1",
      "enabled": true,
      "type": "circle",
      "center": { "x": 300.0, "y": 450.0 },
      "radius": 80.0,
      "z_min": 0.0,
      "z_max": 100.0,
      "safety_margin_m": 15.0
    },
    {
      "id": "obs_r1",
      "enabled": true,
      "type": "rectangle",
      "center": { "x": 700.0, "y": 900.0 },
      "length": 160.0,
      "width": 60.0,
      "yaw_deg": 30.0,
      "z_min": 0.0,
      "z_max": 80.0,
      "safety_margin_m": 10.0
    }
  ]
}
```

说明：
- 当前默认规划器 `comb_no_obstacle` 不会使用障碍物字段。
- 当 `planner.type` 切到 `comb_with_obstacle` 后，规划器将读取这些参数做避障。

## `path.json`
由 `planning.py` 生成，包含：
- 每台 UUV 的 `waypoints`
- 子区域带宽信息
- 路径点数量统计

## `config/network_format.json`
网络发送模板配置，支持：
- `fields`：按顺序定义发送字段（逗号分隔）
- `value_formats`：各字段格式化精度
- `regex_rules`：发送前正则替换规则（可用于帧头帧尾兼容转换）

例如可定义为：
`$,uuv_id,x,y,z,pitch,roll,yaw,u,v,w,p,q,r,time*&&`

---

## 6. 网络发送协议
`network.py` 发送格式固定为：

```text
$uuv_id,x,y,z,pitch,roll,yaw,time*&&
```

示例：

```text
$uuv_1,100.000,200.000,20.000,0.000000,0.000000,1.570796,2.340*&&
```

说明：
- TCP 多进程发送：每台 UUV 一个进程、一个 socket
- 默认目标为 `127.0.0.1:5000+i`
- GUI 中可为每个 UUV 单独设置 `IP/Port`
- 端点也可由 `config/network_endpoints.json` 统一管理

---

## 7. 可视化工具

规划路径图（静态图）：
```bash
python show_planned_paths.py
```

轨迹回放（OpenCV）：
```bash
python show_trajectory_cv.py
```

---

## 8. 常见问题

1. GUI 启动失败提示 `tkinter` 相关错误
- 请确认当前 Python 环境包含 Tk 支持。

2. OpenCV 回放卡顿
- 降低 `--display-fps` 或增大 `--speedup`。

3. 网络端无数据
- 检查接收端 IP/端口是否和 GUI 端点配置一致。
- 确认系统防火墙未阻止 UDP。

---

## 9. 开源协作建议
- 提交 PR 前建议至少执行：
  - `python planning.py`
  - `python motion_server.py --max-rows-per-uuv 100`
  - `python network.py --max-frames 3`
- 保持 `config/init.json`、`path.json` 字段兼容，避免破坏 GUI 读写。

---

## 10. License
见项目根目录 [LICENSE](LICENSE)。
