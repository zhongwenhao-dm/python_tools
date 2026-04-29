#!/usr/bin/env python3
"""
在数据目录中读取 ``flow.txt`` 与 ``imu.txt``，绘制**两条空间轨迹**（均与 IMU 有关）：

1. **光流 + IMU**：光流传感器约定 ``x=左、y=前、z=下``，IMU 约定 ``x=右、y=前、z=上``；
   另加一个 yaw 安装偏差（默认 +10°，使 IMU +y 前进在 flow 系表现为 y 增大、x 减小）。
   先用 IMU 角速度估计相邻 flow 时刻的传感器旋转，在传感器平面中心近似下扣除旋转光流；
   再从首条 **IMU** 起积分姿态，在**首条 flow** 处作姿态锚定 ``R0^{-1}R``，位置从
   **(0,0,0)** 起累加补偿后的每步光流（仍为 flow 原始单位/无量纲，非米）。
2. **纯 IMU 惯导**：**首点位置、速度 0、姿态 I**，再按时间步用 ``a_w = R @ a + g`` 与陀螺积分
   （**米**，**漂移极大**）；对返回再 ``p -= p[0]`` 保证首点为 0。
3. **PRY 曲线**：同一份陀螺积分结果转为 pitch/roll/yaw（deg），直接观察旋转变化。

不绘制无 IMU 的像方像素轨迹。读 ``imu.txt`` 的 ``ax~wz``，**不读**四元数列。

依赖：numpy, scipy, matplotlib
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

try:
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
    raise SystemExit("需要: pip install matplotlib") from e

# 与 imu_lekiwi_fusion 一致：允许从任意 cwd 导入 easy_odom/visual
_EASY_ODOM_ROOT = Path(__file__).resolve().parent.parent
if str(_EASY_ODOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_EASY_ODOM_ROOT))

from pose_fusion import G_WORLD_ENU  # noqa: E402

# 光流传感器坐标：x=左、y=前、z=下；IMU 坐标：x=右、y=前、z=上。
# 正 yaw 偏差定义在 flow 坐标系中：IMU +y 前进映射到 flow 系时 x<0、y>0。
FLOW_YAW_OFFSET_DEG = 10.0
_R_FLOW_TO_IMU_BASE = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)
_R_YAW_IMU_TO_FLOW = Rotation.from_euler("z", FLOW_YAW_OFFSET_DEG, degrees=True).as_matrix()
R_IMU_TO_FLOW = _R_YAW_IMU_TO_FLOW @ _R_FLOW_TO_IMU_BASE.T
# v_imu = R_FLOW_TO_IMU @ v_flow。
R_FLOW_TO_IMU = R_IMU_TO_FLOW.T

# 旋转补偿把 rad 转成 flow 原始单位需要一个等效增益；先按“无尺度”处理为 1。
FLOW_ROT_GAIN_X = 1.0
FLOW_ROT_GAIN_Y = 1.0

# flow.txt: 行1 "--- timestamp: <sec> ---", 行2 为整行 PointStamped
_TS_LINE = re.compile(r"^--- timestamp: ([0-9.]+) ---\s*$")
_POINT = re.compile(
    r"Point\(x=([-\d.eE+]+), y=([-\d.eE+]+), z=([-\d.eE+]+)"
)


def parse_flow_txt(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 (t_recv, fx, fy, fz)，与文件中顺序一致；t_recv 为 parser 打的接收时间戳（秒，浮点）。
    """
    t_list: list[float] = []
    fx_list: list[float] = []
    fy_list: list[float] = []
    fz_list: list[float] = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line0 = lines[i].strip()
        m0 = _TS_LINE.match(line0)
        if m0 and i + 1 < len(lines):
            t_recv = float(m0.group(1))
            m1 = _POINT.search(lines[i + 1])
            if m1:
                t_list.append(t_recv)
                fx_list.append(float(m1.group(1)))
                fy_list.append(float(m1.group(2)))
                fz_list.append(float(m1.group(3)))
            i += 2
            continue
        i += 1

    if not t_list:
        raise ValueError(f"未解析到任何 PointStamped: {path}")

    return (
        np.array(t_list, dtype=np.float64),
        np.array(fx_list, dtype=np.float64),
        np.array(fy_list, dtype=np.float64),
        np.array(fz_list, dtype=np.float64),
    )


def load_imu_txt(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取与 easy_odom 相同的 imu.txt：第 1 行表头。至少 7 列：
    timestamp, ax,ay,az (m/s²), wx,wy,wz (rad/s)。若有四元数列不读取。
    返回 (t, acc( N×3), gyro( N×3))，时间已排序。
    """
    data = np.loadtxt(path, delimiter=",", dtype=np.float64, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 7:
        raise ValueError(
            f"imu.txt 需至少 7 列 (时间+加表+角速度)，当前 {data.shape[1]} 列: {path}"
        )
    t = data[:, 0]
    acc = data[:, 1:4]
    gyro = data[:, 4:7]
    o = np.argsort(t)
    return t[o], acc[o], gyro[o]


def integrate_gyro_to_quaternions(t_imu: np.ndarray, gyro: np.ndarray) -> np.ndarray:
    """
    在机体系用 ``R_{k+1} = R_k * Exp(ω̄ dt)`` 积分，ω̄ 为相邻样点角速度梯形平均（rad/s）。
    第 0 个时刻为单位四元数（世界 = 该时刻 IMU 机体系）。返回 (N,4) 的 xyzw。
    """
    n = len(t_imu)
    if n < 2:
        raise ValueError("IMU 时间序列至少需 2 个样本以积分角速度")
    quat = np.zeros((n, 4), dtype=np.float64)
    quat[0, :] = [0.0, 0.0, 0.0, 1.0]
    R_acc = Rotation.identity()
    for k in range(n - 1):
        dt = float(t_imu[k + 1] - t_imu[k])
        if dt <= 0.0:
            raise ValueError(f"时间须严格递增, 在索引 {k} 处 dt={dt}")
        w = 0.5 * (gyro[k] + gyro[k + 1])
        R_acc = R_acc * Rotation.from_rotvec(w * dt)
        quat[k + 1] = R_acc.as_quat()
    return quat


def integrate_gyro_to_pry_deg(t_imu: np.ndarray, gyro: np.ndarray) -> np.ndarray:
    """
    由角速度积分姿态，并返回 (pitch, roll, yaw) 角度曲线（degree）。
    内部欧拉角按 scipy ``xyz`` 得到 roll/pitch/yaw，再重排为 PRY 便于查看。
    """
    quat = integrate_gyro_to_quaternions(t_imu, gyro)
    rpy = Rotation.from_quat(quat).as_euler("xyz", degrees=True)
    return np.column_stack([rpy[:, 1], rpy[:, 0], rpy[:, 2]])


def integrate_imu_ins_position_enu(
    t_imu: np.ndarray,
    acc: np.ndarray,
    gyro: np.ndarray,
    g_world: np.ndarray,
) -> np.ndarray:
    """
    纯 IMU：与 ``pose_fusion._predict_state_ekf`` 一致，每步先右乘陀螺得 ``R``，
    再 ``a_w = R @ a_b + g``，``v += a_w dt``，``p += v dt + 0.5 a_w dt²``（用步起速度）。
    首时刻位置、速度为 0，首姿态为 I；之后按 IMU 时间步进积分。输出 (N,3) 单位 m，**漂移严重**。
    对返回做 ``p -= p[0]`` 保证首点严格为 0（消除浮点残差）。
    """
    n = len(t_imu)
    if n < 2:
        raise ValueError("IMU 至少 2 个样本")
    p = np.zeros((n, 3), dtype=np.float64)
    v = np.zeros((n, 3), dtype=np.float64)
    R = Rotation.identity()
    g = np.asarray(g_world, dtype=np.float64).ravel()
    for k in range(n - 1):
        dt = float(t_imu[k + 1] - t_imu[k])
        if dt <= 0.0:
            raise ValueError(f"时间须严格递增, 索引 {k} 处 dt={dt}")
        w = 0.5 * (gyro[k] + gyro[k + 1])
        R = R * Rotation.from_rotvec(w * dt)
        a_b = 0.5 * (acc[k] + acc[k + 1])
        a_w = R.apply(a_b) + g
        v[k + 1] = v[k] + a_w * dt
        p[k + 1] = p[k] + v[k] * dt + 0.5 * a_w * (dt**2.0)
    p -= p[0:1, :]
    return p


def compensate_flow_rotation_center(
    fx: np.ndarray,
    fy: np.ndarray,
    rots_imu_to_world_at_flow: Rotation,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    用相邻 flow 时刻之间的 IMU 姿态变化，扣除纯旋转在光流传感器平面中心产生的光流。

    近似假设 flow 是传感器平面中心附近的平均位移。光流传感器坐标为 x=左、y=前、z=下；
    小旋转 ``theta_flow=(rx,ry,rz)`` 在平面中心造成一阶旋转光流：
    ``dx=-gain_x*ry, dy=gain_y*rx``。这里从原始 flow 中减掉该旋转项。
    """
    n = len(fx)
    fx_corr = np.asarray(fx, dtype=np.float64).copy()
    fy_corr = np.asarray(fy, dtype=np.float64).copy()
    if n < 2:
        return fx_corr, fy_corr

    for i in range(n - 1):
        r_delta_imu = (
            rots_imu_to_world_at_flow[i].inv() * rots_imu_to_world_at_flow[i + 1]
        )
        theta_imu = r_delta_imu.as_rotvec()
        theta_flow = R_IMU_TO_FLOW @ theta_imu
        dx_rot = -FLOW_ROT_GAIN_X * theta_flow[1]
        dy_rot = FLOW_ROT_GAIN_Y * theta_flow[0]
        fx_corr[i] -= dx_rot
        fy_corr[i] -= dy_rot
    return fx_corr, fy_corr


def integrate_flow_to_world_path(
    fx: np.ndarray,
    fy: np.ndarray,
    t_flow: np.ndarray,
    t_imu: np.ndarray,
    gyro: np.ndarray,
) -> np.ndarray:
    """
    由角速度从 **首条 IMU** 起积分姿态，Slerp 到各 flow 时刻。先根据相邻 flow 时刻的传感器旋转
    扣除中心近似旋转光流；补偿后的每步 (fx, fy) 按光流坐标 ``x=左、y=前`` 组成平面位移，
    经 ``R_FLOW_TO_IMU`` 到 IMU 系，再变到**以首条 flow 为参考**的世界系：

    令 ``R_i`` 为 Slerp 在 ``t_flow[i]`` 处的姿态，``R0 = R_0``，则
    ``d_i' = (R0^{-1} R_i) @ v_i``，使**首条 flow 处机体系与参考世界系对齐**；位置从
    **(0,0,0)** 起累加：``p[0]=0``，``p[i]=\\sum_{j=0}^{i-1} d_j'``。

    返回 (N, 3)（与 (fx/fx, fy/fy) 同量级，非米）。
    """
    if len(t_imu) < 2:
        raise ValueError("IMU 时间序列至少需要 2 个样本以作插值")
    quat_series = integrate_gyro_to_quaternions(t_imu, gyro)
    t_lo, t_hi = float(t_imu[0]), float(t_imu[-1])
    tq = np.clip(t_flow, t_lo, t_hi)
    if np.any(t_flow < t_lo) or np.any(t_flow > t_hi):
        print("警告: 部分 flow 时间超出 IMU 时间范围，姿态已钳位到 IMU 端点", file=sys.stderr)

    key_rots = Rotation.from_quat(quat_series)
    slerp = Slerp(t_imu, key_rots)
    rots = slerp(tq)

    n = len(fx)
    if n < 1:
        raise ValueError("flow 至少 1 条")
    fx_corr, fy_corr = compensate_flow_rotation_center(fx, fy, rots)
    v_c = np.stack(
        [fx_corr, fy_corr, np.zeros(n, dtype=np.float64)],
        axis=1,
    )
    v_i = (R_FLOW_TO_IMU @ v_c.T).T
    R0 = rots[0]
    d_world = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        d_world[i, :] = (R0.inv() * rots[i]).apply(v_i[i, :])
    # p[0]=0, p[i]=sum(d[0..i-1])，共 n 行
    return np.vstack(
        (np.zeros((1, 3), dtype=np.float64), np.cumsum(d_world, axis=0)[:-1, :])
    )


def plot_trajectory(
    t_flow: np.ndarray,
    fz: np.ndarray,
    world_flow_xyz: np.ndarray,
    p_imu_enu: np.ndarray,
    t_imu: np.ndarray,
    imu_pry_deg: np.ndarray,
    *,
    title: str = "flow + IMU vs pure IMU INS",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    ncols = 4
    fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols + 0.8, 5.0))
    ax0, ax1, ax2, ax3 = axes[0], axes[1], axes[2], axes[3]

    wx, wy = world_flow_xyz[:, 0], world_flow_xyz[:, 1]
    ax0.plot(wx, wy, color="navy", lw=1.2, label="rot-comp flow + ∫ω")
    ax0.scatter(wx[0], wy[0], c="green", s=40, zorder=5, label="start")
    if len(wx) > 1:
        ax0.scatter(wx[-1], wy[-1], c="crimson", s=40, zorder=5, label="end")
    ax0.set_xlabel("cum x (flow units, not m)")
    ax0.set_ylabel("cum y (flow units, not m)")
    ax0.set_title("rotation-compensated flow → world")
    ax0.set_aspect("equal", adjustable="box")
    ax0.grid(True, alpha=0.35)
    ax0.legend(loc="best", fontsize=8)

    px, py = p_imu_enu[:, 0], p_imu_enu[:, 1]
    ax1.plot(px, py, color="darkorange", lw=1.0, label="INS ∫∫(R a+g)")
    ax1.scatter(px[0], py[0], c="green", s=40, zorder=5, label="start")
    if len(px) > 1:
        ax1.scatter(px[-1], py[-1], c="crimson", s=40, zorder=5, label="end")
    ax1.set_xlabel("x (m, ENU, drift)")
    ax1.set_ylabel("y (m, ENU, drift)")
    ax1.set_title("pure IMU: acc+gyro, same g as pose_fusion")
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="best", fontsize=8)

    ax2.plot(t_flow - t_flow[0], fz, color="0.4", lw=0.9)
    ax2.set_xlabel("t − t0 (s)")
    ax2.set_ylabel("z (raw, e.g. feature count)")
    ax2.set_title("flow Point.z (aux)")
    ax2.grid(True, alpha=0.35)

    tt_imu = t_imu - t_imu[0]
    ax3.plot(tt_imu, imu_pry_deg[:, 0], lw=1.0, label="pitch")
    ax3.plot(tt_imu, imu_pry_deg[:, 1], lw=1.0, label="roll")
    ax3.plot(tt_imu, imu_pry_deg[:, 2], lw=1.0, label="yaw")
    ax3.set_xlabel("t − t0 (s)")
    ax3.set_ylabel("angle (deg)")
    ax3.set_title("gyro-integrated PRY")
    ax3.grid(True, alpha=0.35)
    ax3.legend(loc="best", fontsize=8)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    if save_path:
        parent = os.path.dirname(os.path.abspath(save_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图已保存 -> {save_path}")

    if show:
        try:
            plt.show()
        finally:
            plt.close(fig)
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="flow.txt + imu.txt：光流/姿态联合轨迹 与 纯 IMU 惯导 对比"
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="数据目录，须同时含 flow.txt 与 imu.txt",
    )
    parser.add_argument("--save", default="", type=str, help="保存 PNG 路径")
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="不弹窗（仅与 --save 同用）",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"数据目录不存在或不是目录: {data_dir}", file=sys.stderr)
        sys.exit(1)

    path = os.path.join(data_dir, "flow.txt")
    if not os.path.isfile(path):
        print(
            f"未在目录中找到 flow.txt: {data_dir}（本程序固定从该文件读取光流）",
            file=sys.stderr,
        )
        sys.exit(1)

    t, fx, fy, fz = parse_flow_txt(path)
    n = len(t)
    print(
        f"数据目录: {data_dir}\n"
        f"解析 {n} 条 flow, t∈[{t[0]:.3f}, {t[-1]:.3f}] s"
    )

    imu_path = os.path.join(data_dir, "imu.txt")
    if not os.path.isfile(imu_path):
        print(
            f"未在目录中找到 imu.txt（与 flow.txt 同目录，本程序需 IMU 作姿态与惯导）: {imu_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    t_imu, acc, gyro = load_imu_txt(imu_path)
    world_flow_xyz = integrate_flow_to_world_path(fx, fy, t, t_imu, gyro)
    p_ins = integrate_imu_ins_position_enu(t_imu, acc, gyro, G_WORLD_ENU)
    pry_deg = integrate_gyro_to_pry_deg(t_imu, gyro)
    wf = world_flow_xyz[-1]
    p_end = p_ins[-1]
    pry_end = pry_deg[-1]
    print(f"已加载 IMU: {imu_path}（{len(t_imu)} 条）")
    print(
        f"光流+∫ω 累加末点 (flow units): ({wf[0]:.4f}, {wf[1]:.4f}, {wf[2]:.4f})"
    )
    print(
        f"纯 IMU 惯导末位置 (m, ENU, 大漂移可预期): ({p_end[0]:.4f}, {p_end[1]:.4f}, {p_end[2]:.4f})"
    )
    print(
        f"末姿态 PRY (deg): pitch={pry_end[0]:.2f}, roll={pry_end[1]:.2f}, yaw={pry_end[2]:.2f}"
    )

    plot_trajectory(
        t,
        fz,
        world_flow_xyz,
        p_ins,
        t_imu,
        pry_deg,
        title=f"{os.path.basename(data_dir)}: flow+ω vs IMU-INS",
        save_path=args.save.strip() or None,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
