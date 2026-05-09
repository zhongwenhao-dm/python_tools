# easy_odom

IMU + Lekiwi 底盘速度融合（链式 / EKF）、平面底盘航迹推算与可视化。依赖：`numpy`、`scipy`、`matplotlib`（仅可视化）。

## 数据格式

- **`imu.txt`**：第 1 行为表头，从第 2 行起为数值。至少 **7 列**：`timestamp, ax, ay, az, wx, wy, wz`（单位：m/s²、rad/s）。若文件后附 `qx,qy,qz,qw`，融合**不会读取**四元数列。
- **`lekiwi_base_velocity.txt`**：第 1 行为表头，从第 2 行起。**7 列**：`timestamp, lx, ly, lz, ax, ay, az`。融合使用 `lx,ly,lz`；底盘平面 DR 对比使用第 **6** 列（索引 6）作为 **yaw 角速度（deg/s）**。

建议在 `easy_odom` 目录下运行脚本，或从项目根使用 `python easy_odom/xxx.py`（`imu_lekiwi_fusion.py` 会自动把本目录加入 `sys.path`）。

---

## `imu_lekiwi_fusion.py`：融合主程序

在 `--data_dir` 下读取 `imu.txt` 与 `lekiwi_base_velocity.txt`，输出 `fused_pose.csv`（`timestamp,x,y,z,qx,qy,qz,qw`）。

### 基本用法

```bash
cd easy_odom
python imu_lekiwi_fusion.py --data_dir /你的数据目录
```

默认：`--fusion ekf`，输出为 `数据目录/fused_pose.csv`。

### 常用参数

| 参数 | 说明 |
|------|------|
| `--data_dir` | 数据目录（内含 `imu.txt` 与底盘文件） |
| `--imu_file_name` | 默认 `imu.txt` |
| `--odom_file_name` | 默认 `lekiwi_base_velocity.txt` |
| `--fusion ekf` | 扩展卡尔曼：IMU 预测 + 轮速观测（默认） |
| `--fusion chain` | 链式：陀螺积分姿态 + 底盘速度积分 |
| `--out` | 输出 CSV 路径；默认 `data_dir/fused_pose.csv` |
| `--vis` | 融合结束后弹出位姿轨迹图 |
| `--save_fig 路径.png` | 保存位姿图（可不写 `--vis` 仅保存） |
| `--compare_dr` | 额外生成「底盘平面 DR vs 融合」对比图 |
| `--save_compare 路径.png` | 对比图保存路径；默认 `data_dir/fusion_vs_dr_compare.png` |

### EKF 调参（仅 `--fusion ekf` 时）

| 参数 | 说明 |
|------|------|
| `--sigma_wheel` | 轮速观测噪声 σ（m/s）；不传则用 `pose_fusion.EKF_SIGMA_WHEEL`，**略大**通常更平滑 |
| `--wheel_ema` | 轮速机体分量指数平滑系数 `(0,1]`，`1` 不平滑；如 `0.35` 减轻锯齿 |
| `--wheel_update_every N` | 每 **N** 次 IMU 预测后做一次轮速更新；`1` 表示每步都更新 |

### 示例

```bash
# EKF + 弹窗 + 保存对比图
python imu_lekiwi_fusion.py --data_dir /path/to/bag --vis --compare_dr

# 链式融合，指定输出
python imu_lekiwi_fusion.py --data_dir /path/to/bag --fusion chain --out /tmp/out.csv

# EKF：轮速更新稀疏、略平滑轮速
python imu_lekiwi_fusion.py --data_dir /path/to/bag --wheel_update_every 5 --wheel_ema 0.35
```

---

## `pose_visualization.py`：可视化已保存的融合轨迹

读取 `fused_pose.csv`（8 列：时间 + 位置 + 四元数），绘制 3D + XY 俯视图。

```bash
cd easy_odom
python pose_visualization.py /path/to/fused_pose.csv
python pose_visualization.py /path/to/fused_pose.csv --out /path/to/pose.png --no-show
```

| 参数 | 说明 |
|------|------|
| `pose_csv` | `fused_pose.csv` 路径（位置参数） |
| `--no_header` | CSV 无表头时不跳过首行 |
| `--out` | 保存 PNG 路径 |
| `--no-show` | 不弹窗，仅保存（需配合 `--out`） |

---

## `flow/flow_calibration.py`：光流尺度标定

根据多组实验的 `cum_x,cum_y,height_m,distance_m` 估计光流米制尺度，并给出旋转补偿尺度建议值。

```bash
cd easy_odom
python flow/flow_calibration.py
python flow/flow_calibration.py --csv /path/to/calib.csv
```

当前已写入的标定结果：`FLOW_METERS_PER_UNIT = 9.88182550804e-05`，`FLOW_ROT_GAIN_X/Y = 609.955213722`。

---

## `flow/flow_accumulate.py`：光流/IMU 轨迹检查

**只接数据目录**（其下固定为 **`flow.txt`** 与 **`imu.txt`**）。依赖：`numpy`、`scipy`、`matplotlib`。

- 左图：flow 经旋转补偿、米制恢复、杆臂补偿后，在首条 flow 姿态处锚定并从 `(0,0,0)` 累加。
- 中图：纯 IMU 惯导双积分，仅作漂移对比。
- 第三图：`flow` 的 `Point.z` 辅助量；第四图：陀螺积分得到的 `pitch/roll/yaw`。
- 坐标约定：flow `x=左,y=前,z=下`；IMU `x=右,y=前,z=上`；默认 `FLOW_YAW_OFFSET_DEG = 10`。
- 平移外参：`FLOW_TRANSLATION_IMU_TO_FLOW_M = [0.025, -0.090, 0]`（IMU 系，IMU 中心到 flow 中心）。

```bash
cd easy_odom
python flow/flow_accumulate.py /你的数据目录
```

## `flow/flow_imu_fusion.py`：Flow + IMU EKF 融合

读取同一数据目录下的 `imu.txt` 与 `flow.txt`，输出 `flow_imu_fused_pose.csv`，同时导出 `flow_velocity_observations.csv` 用于检查 flow 速度观测。

融合逻辑：

- 开头 `2s` 静止段估计 IMU 零偏；只用 `ax,ay,az,wx,wy,wz`，不读取四元数列。
- IMU 加速度/角速度做预测，flow 旋转补偿 + 米制尺度 + 杆臂补偿后作为机体系 `vx,vy` 观测更新 EKF。
- 默认平面约束：`z = vz = roll = pitch = 0`，只估计平面位置、速度和 yaw。
- 平面 `ax/ay` 降权使用：直行 `ACCEL_XY_SCALE_STRAIGHT = 0.20`，急转弯 `ACCEL_XY_SCALE_TURN = 0.05`（`|wz| >= 0.80 rad/s`）。
- 低速 flow 触发零速更新：`speed <= 0.05 m/s` 使用 `FLOW_ZUPT_SIGMA_VEL = 0.025`。
- 异常观测过滤：`FLOW_MAX_SPEED_MPS = 1.5`，`FLOW_GATE_CHI2 = 5.99`，`FLOW_MIN_FEATURE_COUNT = 20`。

```bash
cd easy_odom
python flow/flow_imu_fusion.py /你的数据目录 --vis
python flow/flow_imu_fusion.py /你的数据目录 --save_fig /tmp/flow_imu.png
python flow/flow_imu_fusion.py /你的数据目录 --flow_vel_out /tmp/flow_vel.csv
```

## `visual/visual_odom.py`：单目视觉里程计

读取 `camera_image_compressed/*.png`，使用 ORB 特征、Essential Matrix 与 `recoverPose` 估计相邻帧相对运动；默认用 `lekiwi_base_velocity.txt` 给单目平移恢复近似尺度。每次运行都会实时显示前后帧匹配/内点追踪和当前累计轨迹。图片相邻帧基线较小时 Essential 容易退化，所以默认 `--step 10` 抽帧处理。

效果不佳...

```bash
cd easy_odom
python visual/visual_odom.py /home/dmgz/ZWH/lerobot/data/lekiwi/my_awesome_kiwi_20260403_161202
python visual/visual_odom.py /你的数据目录 --start_percent 20 --end_percent 60
python visual/visual_odom.py /你的数据目录 --scale_mode unit --max_frames 200
```

实时窗口左侧为前后帧 ORB 匹配/内点追踪，右侧为实时累计 XY 轨迹；可用 `--realtime_matches` 控制左图最多绘制的匹配数量。`--start_percent/--end_percent` 会先按完整图片序列百分比裁剪，再按 `--step` 抽帧。

---

## Python 中直接调用

```python
from pose_fusion import fuse_pose_ekf, fuse_pose_imu_odom

# imu_data / odom_data: numpy 二维数组，格式同上文 CSV（含表头时勿把表头读入）
t, pos, quat = fuse_pose_ekf(imu_data, odom_data)
# t, pos, quat = fuse_pose_imu_odom(imu_data, odom_data)

# EKF 可选关键字参数示例（见 pose_fusion.fuse_pose_ekf 文档字符串）：
# sigma_wheel=..., wheel_meas_ema_alpha=..., wheel_update_every=...
```

底盘平面 DR 辅助：`planar_chassis_dr.compute_chassis_dr_for_fusion_grid(odom_data, t_fused)`。
