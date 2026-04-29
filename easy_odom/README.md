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

## `flow/flow_accumulate.py`：光流 + 纯 IMU 对比

**只接数据目录**（其下 **`flow.txt`** 与 **`imu.txt`** 固定文件名）。依赖：`numpy`、`scipy`、`matplotlib`。

- **不画**「仅像方像素累加、与 IMU 无关」的轨迹；左图、中图**均**由 IMU 参与。
- **左图**：光流传感器坐标按 **`x=左、y=前、z=下`** 处理，不使用 `camera_calibration.py` 的相机外参；默认加 **`FLOW_YAW_OFFSET_DEG = 10`** 的安装 yaw 偏差（IMU +y 前进时 flow 系 `y` 增大、`x` 减小）。先估计并扣除中心近似旋转光流，再在**首条 flow** 处用 `R0^{-1}R` 锚定，位置从 **(0,0,0)** 起累加（flow 原始单位，非米）。
- **中图**：**首点 p、v=0、R=I** 后惯导双积分，且 **p[0] 强置为 0**（**米**，漂移大）。第三图：`flow` 的 `Point.z`。第四图：陀螺积分得到的 **pitch/roll/yaw** 曲线。

```bash
cd easy_odom
python flow/flow_accumulate.py /你的数据目录
```

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
