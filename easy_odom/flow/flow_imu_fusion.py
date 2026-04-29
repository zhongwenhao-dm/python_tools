#!/usr/bin/env python3
"""
Flow + IMU EKF fusion.

输入数据目录固定包含：
- imu.txt: timestamp, ax, ay, az, wx, wy, wz[, ...]
- flow.txt: rosbag_parser 导出的 PointStamped 文本，point.x/point.y 为每周期光流增量

融合思路：
- IMU 只读取 ax,ay,az,wx,wy,wz；不读取、不使用 imu.txt 中可能存在的 qx,qy,qz,qw
- 初始状态固定为 p=0, v=0, 姿态=identity
- 使用开头 2s 估计 IMU 零偏：gyro_bias=mean(gyro)，acc_bias=mean(acc)-[0,0,g]
- IMU 加速度/角速度做高频预测：x=[p(3), v(3), euler_xyz(3)]
- flow 先做旋转光流补偿、米制尺度恢复、杆臂补偿，得到 IMU 机体系速度观测
- EKF 用 body-frame 2D velocity (vx, vy) 更新世界系速度/姿态状态；低速 flow 触发零速更新

所有外参、尺度、噪声和门限都写在本文件顶部。
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


# -----------------------------------------------------------------------------
# Parameters: geometry / calibration
# -----------------------------------------------------------------------------

G_WORLD_ENU = np.array([0.0, 0.0, -9.80665], dtype=np.float64)

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
R_FLOW_TO_IMU = R_IMU_TO_FLOW.T

# flow_calibration.py 标定结果。
FLOW_METERS_PER_UNIT = 9.88182550804e-05
FLOW_METERS_PER_UNIT_PER_METER_HEIGHT = 0.001639464632
FLOW_ROT_GAIN_X = 609.955213722  # flow units / rad
FLOW_ROT_GAIN_Y = 609.955213722  # flow units / rad

# 平移外参：IMU 坐标系下，从 IMU 中心指向 flow 传感器中心（m）。
FLOW_TRANSLATION_IMU_TO_FLOW_M = np.array([0.025, -0.090, 0.0], dtype=np.float64)


# -----------------------------------------------------------------------------
# Parameters: EKF / robustness
# -----------------------------------------------------------------------------

# 初始协方差标准差：p(m), v(m/s), euler(rad)
INIT_SIGMA_POS = 1e-3
INIT_SIGMA_VEL = 0.30
INIT_SIGMA_EULER = 0.10

# 过程噪声标准差：按 dt 缩放加入协方差。
SIGMA_POS_PROCESS = 1e-3
SIGMA_VEL_PROCESS = 0.15
SIGMA_EULER_PROCESS = 0.03

# flow 速度观测噪声（m/s）。实际观测噪声还会按 feature count 调整。
FLOW_SIGMA_VEL_BASE = 0.20
FLOW_FEATURE_REF = 60.0
FLOW_MIN_FEATURE_COUNT = 20.0

# 开头静止标定 IMU bias；要求数据开始阶段小车静止。
IMU_BIAS_ESTIMATION_SEC = 2.0

# 速度和创新门限。
FLOW_MAX_SPEED_MPS = 1.5
FLOW_GATE_CHI2 = 5.99  # 2D measurement, about 95%

# flow 零速检测：低于该速度时，把观测强制设为零速，sigma 更小。
FLOW_ZUPT_SPEED_THRESH_MPS = 0.05
FLOW_ZUPT_SIGMA_VEL = 0.025

# 数值雅可比步长。
JAC_EPS = 1e-6

# 地面小车平面约束：强制 z/vz/roll/pitch 为 0，只保留 x/y/yaw 与平面速度。
ENABLE_PLANAR_CONSTRAINT = True
PLANAR_CONSTRAINT_VARIANCE = 1e-8
PLANAR_STATE_INDICES = (2, 5, 6, 7)  # z, vz, roll, pitch in x=[p,v,euler_xyz]

# 使用前 2s bias 校准后的 ax/ay 做平面 IMU 预测，但降低权重，避免急转弯横向加速度拉飞轨迹。
USE_ACCEL_XY_IN_PREDICTION = True
ACCEL_XY_SCALE_STRAIGHT = 0.20
ACCEL_XY_SCALE_TURN = 0.05
TURN_RATE_THRESH_RAD_S = 0.80


# -----------------------------------------------------------------------------
# Data parsing
# -----------------------------------------------------------------------------

_TS_LINE = re.compile(r"^--- timestamp: ([0-9.]+) ---\s*$")
_POINT = re.compile(
    r"Point\(x=([-\d.eE+]+), y=([-\d.eE+]+), z=([-\d.eE+]+)"
)


@dataclass(frozen=True)
class FlowVelocityMeasurement:
    timestamp: float
    vel_body_xy: np.ndarray
    feature_count: float
    dt: float


@dataclass
class FusionStats:
    flow_total: int = 0
    flow_used: int = 0
    flow_zupt_used: int = 0
    flow_skipped_feature: int = 0
    flow_skipped_speed: int = 0
    flow_skipped_gate: int = 0
    flow_skipped_time: int = 0


def parse_flow_txt(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (timestamp, fx, fy, fz)。fx/fy 为每周期光流增量，fz 常为特征数。"""
    t_list: list[float] = []
    fx_list: list[float] = []
    fy_list: list[float] = []
    fz_list: list[float] = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        m0 = _TS_LINE.match(lines[i].strip())
        if m0 and i + 1 < len(lines):
            m1 = _POINT.search(lines[i + 1])
            if m1:
                t_list.append(float(m0.group(1)))
                fx_list.append(float(m1.group(1)))
                fy_list.append(float(m1.group(2)))
                fz_list.append(float(m1.group(3)))
            i += 2
            continue
        i += 1

    if not t_list:
        raise ValueError(f"未解析到任何 flow PointStamped: {path}")
    return (
        np.array(t_list, dtype=np.float64),
        np.array(fx_list, dtype=np.float64),
        np.array(fy_list, dtype=np.float64),
        np.array(fz_list, dtype=np.float64),
    )


def load_imu_txt(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 imu.txt，返回 (t, acc, gyro)，忽略可能存在的四元数列。"""
    data = np.loadtxt(path, delimiter=",", dtype=np.float64, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 7:
        raise ValueError(f"imu.txt 至少需要 7 列，当前 {data.shape[1]} 列: {path}")
    t = data[:, 0]
    acc = data[:, 1:4]
    gyro = data[:, 4:7]
    order = np.argsort(t)
    return t[order], acc[order], gyro[order]


def estimate_imu_biases(
    t_imu: np.ndarray,
    acc: np.ndarray,
    gyro: np.ndarray,
    duration_sec: float = IMU_BIAS_ESTIMATION_SEC,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用开头静止段估计 IMU bias。

    初始姿态固定 identity，静止时期望 acc≈[0,0,+g]，所以：
    acc_bias = mean(acc) - [0,0,+g]
    gyro_bias = mean(gyro)
    """
    if len(t_imu) == 0:
        raise ValueError("IMU 数据为空")
    t0 = float(t_imu[0])
    mask = t_imu <= t0 + float(duration_sec)
    if int(np.sum(mask)) < 2:
        raise ValueError(f"用于 bias 估计的 IMU 样本过少: {int(np.sum(mask))}")
    mean_acc = np.mean(acc[mask], axis=0)
    mean_gyro = np.mean(gyro[mask], axis=0)
    expected_static_acc = np.array([0.0, 0.0, -G_WORLD_ENU[2]], dtype=np.float64)
    acc_bias = mean_acc - expected_static_acc
    gyro_bias = mean_gyro
    return acc_bias, gyro_bias


# -----------------------------------------------------------------------------
# Flow preprocessing
# -----------------------------------------------------------------------------

def integrate_gyro_to_quaternions(t_imu: np.ndarray, gyro: np.ndarray) -> np.ndarray:
    """右乘 Exp(omega*dt) 积分陀螺，返回 scipy xyzw 四元数。"""
    n = len(t_imu)
    if n < 2:
        raise ValueError("IMU 至少需要 2 个样本")
    quat = np.zeros((n, 4), dtype=np.float64)
    quat[0] = [0.0, 0.0, 0.0, 1.0]
    r_acc = Rotation.identity()
    for i in range(n - 1):
        dt = float(t_imu[i + 1] - t_imu[i])
        if dt <= 0.0:
            raise ValueError(f"IMU 时间须严格递增，索引 {i} 处 dt={dt}")
        w = 0.5 * (gyro[i] + gyro[i + 1])
        r_acc = r_acc * Rotation.from_rotvec(w * dt)
        quat[i + 1] = r_acc.as_quat()
    return quat


def slerp_imu_rotations_at(t_query: np.ndarray, t_imu: np.ndarray, gyro: np.ndarray) -> Rotation:
    quat = integrate_gyro_to_quaternions(t_imu, gyro)
    tq = np.clip(t_query, float(t_imu[0]), float(t_imu[-1]))
    return Slerp(t_imu, Rotation.from_quat(quat))(tq)


def compensate_flow_rotation_center(
    fx: np.ndarray,
    fy: np.ndarray,
    rots_imu_to_world_at_flow: Rotation,
) -> Tuple[np.ndarray, np.ndarray]:
    """中心近似旋转光流补偿，输出仍为 flow units。"""
    n = len(fx)
    fx_corr = np.asarray(fx, dtype=np.float64).copy()
    fy_corr = np.asarray(fy, dtype=np.float64).copy()
    for i in range(max(0, n - 1)):
        r_delta_imu = rots_imu_to_world_at_flow[i].inv() * rots_imu_to_world_at_flow[i + 1]
        theta_flow = R_IMU_TO_FLOW @ r_delta_imu.as_rotvec()
        dx_rot = -FLOW_ROT_GAIN_X * theta_flow[1]
        dy_rot = FLOW_ROT_GAIN_Y * theta_flow[0]
        fx_corr[i] -= dx_rot
        fy_corr[i] -= dy_rot
    return fx_corr, fy_corr


def compute_lever_arm_delta_imu_m(rots_imu_to_world_at_flow: Rotation) -> np.ndarray:
    """每个 flow 周期内，由杆臂旋转造成的 flow 中心相对 IMU 中心位移（IMU 系，m）。"""
    n = len(rots_imu_to_world_at_flow)
    out = np.zeros((n, 3), dtype=np.float64)
    r = FLOW_TRANSLATION_IMU_TO_FLOW_M
    for i in range(max(0, n - 1)):
        r_delta_imu = rots_imu_to_world_at_flow[i].inv() * rots_imu_to_world_at_flow[i + 1]
        out[i] = r_delta_imu.apply(r) - r
    return out


def build_flow_velocity_measurements(
    t_flow: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    fz: np.ndarray,
    t_imu: np.ndarray,
    gyro: np.ndarray,
) -> list[FlowVelocityMeasurement]:
    """
    将 flow 增量转换为 IMU 机体系速度观测 (vx, vy)。

    第 i 条 flow 增量视为 [t_flow[i], t_flow[i+1]] 区间位移，观测时间放在 t_flow[i+1]。
    """
    if len(t_flow) < 2:
        return []
    rots = slerp_imu_rotations_at(t_flow, t_imu, gyro)
    fx_corr, fy_corr = compensate_flow_rotation_center(fx, fy, rots)
    lever_delta = compute_lever_arm_delta_imu_m(rots)

    measurements: list[FlowVelocityMeasurement] = []
    for i in range(len(t_flow) - 1):
        dt = float(t_flow[i + 1] - t_flow[i])
        if dt <= 0.0:
            continue
        delta_flow_m = R_FLOW_TO_IMU @ np.array([fx_corr[i], fy_corr[i], 0.0], dtype=np.float64)
        delta_body_m = delta_flow_m * FLOW_METERS_PER_UNIT - lever_delta[i]
        vel_body = delta_body_m / dt
        measurements.append(
            FlowVelocityMeasurement(
                timestamp=float(t_flow[i + 1]),
                vel_body_xy=vel_body[:2].copy(),
                feature_count=float(fz[i]),
                dt=dt,
            )
        )
    return measurements


# -----------------------------------------------------------------------------
# EKF
# -----------------------------------------------------------------------------

def predict_state(x: np.ndarray, accel_b: np.ndarray, gyro_b: np.ndarray, dt: float) -> np.ndarray:
    """x=[p(3),v(3),euler_xyz(3)]。"""
    p = x[0:3]
    v = x[3:6]
    euler = x[6:9]

    if ENABLE_PLANAR_CONSTRAINT and not USE_ACCEL_XY_IN_PREDICTION:
        yaw_new = float(euler[2] + gyro_b[2] * dt)
        p_new = p + v * dt
        v_new = v.copy()
        euler_new = np.array([0.0, 0.0, yaw_new], dtype=np.float64)
        p_new[2] = 0.0
        v_new[2] = 0.0
        return np.concatenate([p_new, v_new, euler_new])

    accel_used = np.asarray(accel_b, dtype=np.float64).copy()
    if ENABLE_PLANAR_CONSTRAINT and USE_ACCEL_XY_IN_PREDICTION:
        scale = (
            ACCEL_XY_SCALE_TURN
            if abs(float(gyro_b[2])) >= TURN_RATE_THRESH_RAD_S
            else ACCEL_XY_SCALE_STRAIGHT
        )
        accel_used[0:2] *= scale

    r = Rotation.from_euler("xyz", euler, degrees=False)
    r_new = r * Rotation.from_rotvec(np.asarray(gyro_b, dtype=np.float64) * dt)
    a_w = r_new.apply(accel_used) + G_WORLD_ENU
    p_new = p + v * dt + 0.5 * a_w * (dt**2)
    v_new = v + a_w * dt
    euler_new = r_new.as_euler("xyz", degrees=False)
    return np.concatenate([p_new, v_new, euler_new])


def numerical_jacobian_predict(
    x: np.ndarray,
    accel_b: np.ndarray,
    gyro_b: np.ndarray,
    dt: float,
    eps: float = JAC_EPS,
) -> np.ndarray:
    x0 = predict_state(x, accel_b, gyro_b, dt)
    f = np.zeros((9, 9), dtype=np.float64)
    for j in range(9):
        dx = np.zeros(9, dtype=np.float64)
        dx[j] = eps
        f[:, j] = (predict_state(x + dx, accel_b, gyro_b, dt) - x0) / eps
    return f


def measure_body_velocity_xy(x: np.ndarray) -> np.ndarray:
    v_w = x[3:6]
    r = Rotation.from_euler("xyz", x[6:9], degrees=False)
    v_b = r.inv().apply(v_w)
    return v_b[:2]


def numerical_jacobian_measure(x: np.ndarray, eps: float = JAC_EPS) -> np.ndarray:
    h0 = measure_body_velocity_xy(x)
    h = np.zeros((2, 9), dtype=np.float64)
    for j in range(9):
        dx = np.zeros(9, dtype=np.float64)
        dx[j] = eps
        h[:, j] = (measure_body_velocity_xy(x + dx) - h0) / eps
    return h


class FlowImuEkf:
    def __init__(self) -> None:
        self.x = np.zeros(9, dtype=np.float64)
        sigma0 = np.array(
            [INIT_SIGMA_POS] * 3 + [INIT_SIGMA_VEL] * 3 + [INIT_SIGMA_EULER] * 3,
            dtype=np.float64,
        )
        self.p = np.diag(sigma0**2)
        self._apply_planar_constraint()

    def _apply_planar_constraint(self) -> None:
        """地面小车约束：z=vz=roll=pitch=0，并收紧对应协方差。"""
        if not ENABLE_PLANAR_CONSTRAINT:
            return
        self.x[2] = 0.0
        self.x[5] = 0.0
        self.x[6] = 0.0
        self.x[7] = 0.0
        for idx in PLANAR_STATE_INDICES:
            self.p[idx, :] = 0.0
            self.p[:, idx] = 0.0
            self.p[idx, idx] = PLANAR_CONSTRAINT_VARIANCE

    def predict(self, accel_b: np.ndarray, gyro_b: np.ndarray, dt: float) -> None:
        f = numerical_jacobian_predict(self.x, accel_b, gyro_b, dt)
        self.x = predict_state(self.x, accel_b, gyro_b, dt)
        sigma_q = np.array(
            [SIGMA_POS_PROCESS] * 3 + [SIGMA_VEL_PROCESS] * 3 + [SIGMA_EULER_PROCESS] * 3,
            dtype=np.float64,
        )
        q = np.diag((sigma_q**2) * max(float(dt), 1e-4))
        self.p = f @ self.p @ f.T + q
        self.p = 0.5 * (self.p + self.p.T)
        self._apply_planar_constraint()

    def update_flow_velocity(self, z_body_xy: np.ndarray, sigma_vel: float) -> tuple[bool, float]:
        z = np.asarray(z_body_xy, dtype=np.float64).ravel()
        h = measure_body_velocity_xy(self.x)
        residual = z - h
        h_jac = numerical_jacobian_measure(self.x)
        r_cov = np.eye(2, dtype=np.float64) * (float(sigma_vel) ** 2)
        s = h_jac @ self.p @ h_jac.T + r_cov
        try:
            s_inv = np.linalg.inv(s)
        except np.linalg.LinAlgError:
            return False, float("inf")
        nis = float(residual.T @ s_inv @ residual)
        if nis > FLOW_GATE_CHI2:
            return False, nis
        k = self.p @ h_jac.T @ s_inv
        self.x = self.x + k @ residual
        i = np.eye(9, dtype=np.float64)
        # Joseph form, more stable.
        self.p = (i - k @ h_jac) @ self.p @ (i - k @ h_jac).T + k @ r_cov @ k.T
        self.p = 0.5 * (self.p + self.p.T)
        self._apply_planar_constraint()
        return True, nis


def sigma_flow_from_feature_count(feature_count: float) -> float:
    c = max(float(feature_count), 1.0)
    return FLOW_SIGMA_VEL_BASE * np.sqrt(FLOW_FEATURE_REF / c)


def fuse_flow_imu(
    t_imu: np.ndarray,
    acc: np.ndarray,
    gyro: np.ndarray,
    flow_measurements: list[FlowVelocityMeasurement],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, FusionStats]:
    stats = FusionStats(flow_total=len(flow_measurements))
    ekf = FlowImuEkf()
    flow_idx = 0
    n_flow = len(flow_measurements)

    out_t = [float(t_imu[0])]
    out_pos = [ekf.x[0:3].copy()]
    out_quat = [Rotation.from_euler("xyz", ekf.x[6:9]).as_quat()]

    for i in range(len(t_imu) - 1):
        dt = float(t_imu[i + 1] - t_imu[i])
        if dt <= 0.0:
            raise ValueError(f"IMU 时间须严格递增，索引 {i} 处 dt={dt}")
        accel_b = 0.5 * (acc[i] + acc[i + 1])
        gyro_b = 0.5 * (gyro[i] + gyro[i + 1])
        ekf.predict(accel_b, gyro_b, dt)

        t_now = float(t_imu[i + 1])
        while flow_idx < n_flow and flow_measurements[flow_idx].timestamp <= t_now:
            m = flow_measurements[flow_idx]
            if m.timestamp < t_imu[0] or m.timestamp > t_imu[-1]:
                stats.flow_skipped_time += 1
                flow_idx += 1
                continue
            if m.feature_count < FLOW_MIN_FEATURE_COUNT:
                stats.flow_skipped_feature += 1
                flow_idx += 1
                continue
            speed = float(np.linalg.norm(m.vel_body_xy))
            if not np.isfinite(speed) or speed > FLOW_MAX_SPEED_MPS:
                stats.flow_skipped_speed += 1
                flow_idx += 1
                continue
            if speed <= FLOW_ZUPT_SPEED_THRESH_MPS:
                ok, _nis = ekf.update_flow_velocity(
                    np.zeros(2, dtype=np.float64),
                    FLOW_ZUPT_SIGMA_VEL,
                )
                if ok:
                    stats.flow_used += 1
                    stats.flow_zupt_used += 1
                else:
                    stats.flow_skipped_gate += 1
                flow_idx += 1
                continue
            ok, _nis = ekf.update_flow_velocity(
                m.vel_body_xy,
                sigma_flow_from_feature_count(m.feature_count),
            )
            if ok:
                stats.flow_used += 1
            else:
                stats.flow_skipped_gate += 1
            flow_idx += 1

        out_t.append(t_now)
        out_pos.append(ekf.x[0:3].copy())
        out_quat.append(Rotation.from_euler("xyz", ekf.x[6:9]).as_quat())

    return (
        np.asarray(out_t, dtype=np.float64),
        np.asarray(out_pos, dtype=np.float64),
        np.asarray(out_quat, dtype=np.float64),
        stats,
    )


def save_pose_csv(path: str, t: np.ndarray, pos: np.ndarray, quat_xyzw: np.ndarray) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    data = np.column_stack([t, pos, quat_xyzw])
    header = "timestamp,x,y,z,qx,qy,qz,qw"
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def save_flow_velocity_csv(path: str, measurements: list[FlowVelocityMeasurement]) -> None:
    """导出每条 flow 速度观测，便于检查异常速度。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    rows = []
    for i, m in enumerate(measurements):
        vx, vy = float(m.vel_body_xy[0]), float(m.vel_body_xy[1])
        speed = float(np.hypot(vx, vy))
        sigma = sigma_flow_from_feature_count(m.feature_count)
        feature_ok = int(m.feature_count >= FLOW_MIN_FEATURE_COUNT)
        speed_ok = int(np.isfinite(speed) and speed <= FLOW_MAX_SPEED_MPS)
        zupt_candidate = int(
            feature_ok and speed_ok and speed <= FLOW_ZUPT_SPEED_THRESH_MPS
        )
        rows.append(
            [
                i,
                m.timestamp,
                m.dt,
                vx,
                vy,
                speed,
                m.feature_count,
                sigma,
                feature_ok,
                speed_ok,
                zupt_candidate,
            ]
        )
    header = (
        "index,timestamp,dt,vx_body_mps,vy_body_mps,speed_mps,"
        "feature_count,sigma_vel_mps,feature_ok,speed_ok,zupt_candidate"
    )
    if rows:
        data = np.asarray(rows, dtype=np.float64)
        np.savetxt(path, data, delimiter=",", header=header, comments="")
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + "\n")


def print_flow_velocity_summary(measurements: list[FlowVelocityMeasurement], top_k: int = 10) -> None:
    if not measurements:
        print("flow velocity observations: none")
        return
    speeds = np.array([np.linalg.norm(m.vel_body_xy) for m in measurements], dtype=np.float64)
    features = np.array([m.feature_count for m in measurements], dtype=np.float64)
    dts = np.array([m.dt for m in measurements], dtype=np.float64)
    too_fast = speeds > FLOW_MAX_SPEED_MPS
    low_feature = features < FLOW_MIN_FEATURE_COUNT
    zupt = (speeds <= FLOW_ZUPT_SPEED_THRESH_MPS) & (~too_fast) & (~low_feature)
    print(
        "flow velocity stats (m/s): "
        f"min={np.min(speeds):.4f}, p50={np.percentile(speeds, 50):.4f}, "
        f"p90={np.percentile(speeds, 90):.4f}, p99={np.percentile(speeds, 99):.4f}, "
        f"max={np.max(speeds):.4f}"
    )
    print(
        "flow observation quality: "
        f"dt[min/median/max]={np.min(dts):.4f}/{np.median(dts):.4f}/{np.max(dts):.4f}s, "
        f"feature[min/median/max]={np.min(features):.1f}/{np.median(features):.1f}/{np.max(features):.1f}, "
        f"too_fast={int(np.sum(too_fast))}, low_feature={int(np.sum(low_feature))}, "
        f"zupt_candidate={int(np.sum(zupt))}"
    )
    order = np.argsort(-speeds)[: min(top_k, len(measurements))]
    print(f"top {len(order)} fastest flow velocity observations:")
    for idx in order:
        m = measurements[int(idx)]
        vx, vy = m.vel_body_xy
        flags = []
        if too_fast[idx]:
            flags.append("too_fast")
        if low_feature[idx]:
            flags.append("low_feature")
        if zupt[idx]:
            flags.append("zupt")
        flag_str = ",".join(flags) if flags else "ok"
        print(
            f"  #{int(idx):04d} t={m.timestamp:.3f} dt={m.dt:.4f} "
            f"v=({vx:.4f},{vy:.4f}) speed={speeds[idx]:.4f} "
            f"feature={m.feature_count:.1f} [{flag_str}]"
        )


def maybe_plot_xy(t: np.ndarray, pos: np.ndarray, save_path: str, show: bool) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise SystemExit("需要 matplotlib 才能画图: pip install matplotlib") from e

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    ax0, ax1 = axes
    ax0.plot(pos[:, 0], pos[:, 1], lw=1.2, color="C0", label="flow+IMU EKF")
    ax0.scatter(pos[0, 0], pos[0, 1], c="green", s=40, label="start", zorder=5)
    ax0.scatter(pos[-1, 0], pos[-1, 1], c="crimson", s=40, label="end", zorder=5)
    ax0.set_xlabel("x (m)")
    ax0.set_ylabel("y (m)")
    ax0.set_title("XY trajectory")
    ax0.set_aspect("equal", adjustable="box")
    ax0.grid(True, alpha=0.35)
    ax0.legend(loc="best", fontsize=8)

    ax1.plot(t - t[0], pos[:, 0], label="x")
    ax1.plot(t - t[0], pos[:, 1], label="y")
    ax1.plot(t - t[0], pos[:, 2], label="z")
    ax1.set_xlabel("t - t0 (s)")
    ax1.set_ylabel("position (m)")
    ax1.set_title("position vs time")
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="best", fontsize=8)

    fig.tight_layout()
    if save_path:
        parent = os.path.dirname(os.path.abspath(save_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图已保存 -> {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="flow.txt + imu.txt EKF 融合")
    parser.add_argument("data_dir", type=str, help="数据目录，须同时含 imu.txt 与 flow.txt")
    parser.add_argument("--out", default="", help="输出 CSV，默认 data_dir/flow_imu_fused_pose.csv")
    parser.add_argument(
        "--flow_vel_out",
        default="",
        help="导出 flow 速度观测 CSV，默认 data_dir/flow_velocity_observations.csv",
    )
    parser.add_argument("--save_fig", default="", help="保存融合轨迹 PNG")
    parser.add_argument("--vis", action="store_true", help="显示融合轨迹图")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    imu_path = os.path.join(data_dir, "imu.txt")
    flow_path = os.path.join(data_dir, "flow.txt")
    if not os.path.isdir(data_dir):
        print(f"数据目录不存在: {data_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(imu_path):
        print(f"未找到 imu.txt: {imu_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(flow_path):
        print(f"未找到 flow.txt: {flow_path}", file=sys.stderr)
        sys.exit(1)

    t_imu, acc_raw, gyro_raw = load_imu_txt(imu_path)
    acc_bias, gyro_bias = estimate_imu_biases(t_imu, acc_raw, gyro_raw)
    acc = acc_raw - acc_bias
    gyro = gyro_raw - gyro_bias
    t_flow, fx, fy, fz = parse_flow_txt(flow_path)
    measurements = build_flow_velocity_measurements(t_flow, fx, fy, fz, t_imu, gyro)
    flow_vel_out = args.flow_vel_out or os.path.join(data_dir, "flow_velocity_observations.csv")
    save_flow_velocity_csv(flow_vel_out, measurements)
    print(
        f"imu={len(t_imu)} samples, flow={len(t_flow)} samples, "
        f"flow velocity measurements={len(measurements)}"
    )
    print("IMU prediction uses only ax,ay,az,wx,wy,wz; qx,qy,qz,qw are ignored if present.")
    print("Initial state: p=0, v=0, attitude=identity.")
    print(
        f"IMU bias estimated from first {IMU_BIAS_ESTIMATION_SEC:.2f}s: "
        f"acc_bias={acc_bias}, gyro_bias={gyro_bias}"
    )
    print(
        f"Flow ZUPT: speed <= {FLOW_ZUPT_SPEED_THRESH_MPS:.3f} m/s -> "
        f"zero velocity update with sigma={FLOW_ZUPT_SIGMA_VEL:.3f} m/s"
    )
    print(f"flow velocity observations written -> {flow_vel_out}")
    print_flow_velocity_summary(measurements)

    t, pos, quat, stats = fuse_flow_imu(t_imu, acc, gyro, measurements)
    out_path = args.out or os.path.join(data_dir, "flow_imu_fused_pose.csv")
    save_pose_csv(out_path, t, pos, quat)
    print(f"已写入 {len(t)} 帧 -> {out_path}")
    print(f"末位置 [x,y,z] m: {pos[-1]}")
    print(
        "flow updates: "
        f"total={stats.flow_total}, used={stats.flow_used}, zupt_used={stats.flow_zupt_used}, "
        f"feature_skip={stats.flow_skipped_feature}, speed_skip={stats.flow_skipped_speed}, "
        f"gate_skip={stats.flow_skipped_gate}, time_skip={stats.flow_skipped_time}"
    )

    if args.vis or args.save_fig:
        maybe_plot_xy(t, pos, args.save_fig, args.vis)


if __name__ == "__main__":
    main()
