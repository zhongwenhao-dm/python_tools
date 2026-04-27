"""
IMU + 底盘数据融合：链式积分与扩展卡尔曼滤波（EKF）。

世界系 ENU、z 向上；重力 g = [0,0,-9.80665] m/s²。
融合仅使用 IMU 加计与陀螺；姿态由陀螺积分（链式）或 EKF 状态估计，不读取 IMU 四元数列。
"""
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

# 世界系 ENU，z 轴向上（与常见 IMU 静止时 az≈+g 一致）
G_WORLD_ENU = np.array([0.0, 0.0, -9.80665], dtype=np.float64)

# EKF 默认噪声与初始协方差对角（全在此集中修改）
# 轮速观测：过小会紧贴带噪轮速 → 轨迹锯齿；略大则更平滑（更信 IMU 预测）。
EKF_SIGMA_WHEEL = 0.3  # 底盘机体线速度观测标准差 (m/s)
EKF_SIGMA_POS_PROCESS = 1e-3  # 过程噪声：位置相关
EKF_SIGMA_VEL_PROCESS = 0.05  # 过程噪声：速度相关
EKF_SIGMA_EULER_PROCESS = 0.02  # 过程噪声：姿态 (rad)
EKF_P_INIT_DIAG = np.array(
    [1e-4, 1e-4, 1e-4, 0.5, 0.5, 0.5, 0.2, 0.2, 0.3],
    dtype=np.float64,
)  # 初始状态协方差对角 σ（位置、速度、欧拉）

# imu.txt: timestamp,ax,ay,az,wx,wy,wz[,qx,qy,qz,qw 可选；融合不使用四元数列]
IMU_COL_TS = 0
IMU_COL_ACC = slice(1, 4)  # ax, ay, az (m/s^2)
IMU_COL_GYRO = slice(4, 7)  # wx, wy, wz (rad/s)
IMU_COL_QUAT = slice(7, 11)  # 若存在则供外部读取；fuse_* 仅使用加计+陀螺
IMU_MIN_COLS = 7  # 融合最少列数：时间 + acc + gyro

# lekiwi_base_velocity.txt: timestamp,lx,ly,lz,ax,ay,az
ODOM_COL_TS = 0
ODOM_COL_LXYZ = slice(1, 4)  # lx, ly, lz（按机体线速度使用）
ODOM_COL_AXYZ = slice(4, 7)  # ax, ay, az（未用于当前 EKF，保留供扩展）

# 底盘速度列向量 v_base = [lx, ly, lz]^T 与 IMU 机体系对齐：v_imu = R_BASE_TO_IMU @ v_base
# 默认：底盘右手系 x=前、y=左、z=上；IMU 固连为 y=前、x=右、z=上 → 绕 +z 转 +90°
# 若安装不同，请只改此矩阵。
R_BASE_TO_IMU = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

__all__ = [
    "G_WORLD_ENU",
    "EKF_SIGMA_WHEEL",
    "EKF_SIGMA_POS_PROCESS",
    "EKF_SIGMA_VEL_PROCESS",
    "EKF_SIGMA_EULER_PROCESS",
    "EKF_P_INIT_DIAG",
    "R_BASE_TO_IMU",
    "base_velocity_to_imu_body",
    "IMU_COL_TS",
    "IMU_COL_ACC",
    "IMU_COL_GYRO",
    "IMU_COL_QUAT",
    "IMU_MIN_COLS",
    "ODOM_COL_TS",
    "ODOM_COL_LXYZ",
    "ODOM_COL_AXYZ",
    "ensure_2d",
    "align_quaternion_continuous",
    "fuse_pose_imu_odom",
    "fuse_pose_ekf",
    "interp_linear_1d",
    "EkfImuWheelFilter",
]


def base_velocity_to_imu_body(v_base: np.ndarray) -> np.ndarray:
    """
    将底盘 CSV 中的 [lx, ly, lz] 变到与 IMU 加速度/陀螺同一机体系（固定 R_BASE_TO_IMU）。
    行向量形式：v_imu_row = v_base_row @ R^T，等价于列向量 v_imu = R @ v_base。
    """
    r = R_BASE_TO_IMU
    v = np.asarray(v_base, dtype=np.float64)
    if v.ndim == 1:
        return v @ r.T
    return v @ r.T


def ensure_2d(a: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        return a.reshape(1, -1)
    return a


def _quat_identity_xyzw() -> np.ndarray:
    """scipy: [qx,qy,qz,qw], identity w=1."""
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def _propagate_quat_gyro(q: np.ndarray, gyro_b: np.ndarray, dt: float) -> np.ndarray:
    """q_{k+1} = q_k * dq(gyro*dt)，归一化。"""
    dq = Rotation.from_rotvec(np.asarray(gyro_b, dtype=np.float64) * float(dt))
    q_new = (Rotation.from_quat(q) * dq).as_quat()
    nrm = float(np.linalg.norm(q_new))
    if nrm < 1e-12:
        return q
    return q_new / nrm


def align_quaternion_continuous(q: np.ndarray) -> np.ndarray:
    """相邻帧四元数点积为负时翻转符号，避免可视化/插值时走大弧。"""
    out = q.copy()
    for i in range(1, len(out)):
        if np.dot(out[i], out[i - 1]) < 0.0:
            out[i] = -out[i]
    return out


def interp_linear_1d(
    t_target: np.ndarray,
    t_src: np.ndarray,
    vals: np.ndarray,
) -> np.ndarray:
    """
    对单通道量按时间线性插值到 t_target；t_src 可乱序，重复时间戳取平均。
    """
    ts = np.asarray(t_src, dtype=np.float64).ravel()
    v = np.asarray(vals, dtype=np.float64).ravel()
    order = np.argsort(ts)
    ts = ts[order]
    v = v[order]
    uniq_ts = np.unique(ts)
    if len(uniq_ts) < len(ts):
        v_uniq = np.empty(len(uniq_ts), dtype=np.float64)
        for k, u in enumerate(uniq_ts):
            v_uniq[k] = v[ts == u].mean()
        ts = uniq_ts
        v = v_uniq
    return np.interp(t_target, ts, v)


def _interp_linear_xyz(
    t_target: np.ndarray,
    t_src: np.ndarray,
    xyz_src: np.ndarray,
) -> np.ndarray:
    v = np.asarray(xyz_src, dtype=np.float64)
    if v.ndim == 1:
        v = v.reshape(-1, 3)
    return np.column_stack(
        [
            interp_linear_1d(t_target, t_src, v[:, 0]),
            interp_linear_1d(t_target, t_src, v[:, 1]),
            interp_linear_1d(t_target, t_src, v[:, 2]),
        ]
    )


def fuse_pose_imu_odom(
    imu_data: np.ndarray,
    odom_data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    链式融合：姿态仅由陀螺积分（首帧单位阵）；位置 = 底盘机体速度经 R_BASE_TO_IMU 对齐后、
    姿态转到世界系后积分。不使用 IMU 四元数列。

    返回 (t, position_w, quaternion_xyzw)，均在 IMU 时间重叠子序列上；
    起点位置为世界系原点；输出四元数为陀螺递推结果（供可视化）。
    """
    imu_data = ensure_2d(imu_data)
    odom_data = ensure_2d(odom_data)
    if imu_data.shape[1] < IMU_MIN_COLS:
        raise ValueError(f"IMU 至少需要 {IMU_MIN_COLS} 列（时间+加计+陀螺），当前 {imu_data.shape[1]}")

    t_imu = imu_data[:, IMU_COL_TS]
    t_odom = odom_data[:, ODOM_COL_TS]
    t0 = max(float(t_imu.min()), float(t_odom.min()))
    t1 = min(float(t_imu.max()), float(t_odom.max()))
    if not (t0 < t1):
        raise ValueError(f"IMU 与底盘时间无有效重叠: imu [{t_imu.min():.6f},{t_imu.max():.6f}], "
                         f"odom [{t_odom.min():.6f},{t_odom.max():.6f}]")

    mask = (t_imu >= t0) & (t_imu <= t1)
    if not np.any(mask):
        raise ValueError("重叠区间内无 IMU 样本")

    t = t_imu[mask]
    imu_s = imu_data[mask]
    gyro = imu_s[:, IMU_COL_GYRO]

    v_base = _interp_linear_xyz(t, t_odom, odom_data[:, ODOM_COL_LXYZ])
    v_body = base_velocity_to_imu_body(v_base)

    n = len(t)
    quat = np.zeros((n, 4), dtype=np.float64)
    quat[0] = _quat_identity_xyzw()
    pos = np.zeros((n, 3), dtype=np.float64)
    for i in range(n - 1):
        dt = float(t[i + 1] - t[i])
        if dt <= 0.0:
            raise ValueError(f"时间须严格递增，在索引 {i} 处 dt={dt}")
        v_w = Rotation.from_quat(quat[i]).apply(v_body[i])
        pos[i + 1] = pos[i] + v_w * dt
        quat[i + 1] = _propagate_quat_gyro(quat[i], gyro[i], dt)

    quat = align_quaternion_continuous(quat)
    return t, pos, quat


def _predict_state_ekf(
    x: np.ndarray,
    accel_b: np.ndarray,
    gyro_b: np.ndarray,
    dt: float,
    g_world: np.ndarray,
) -> np.ndarray:
    """状态 x=[p(3),v(3),euler_xyz(3)]，陀螺右乘积分四元数，加计转世界系积分 v、p。"""
    p, v, euler = x[0:3], x[3:6], x[6:9]
    q = Rotation.from_euler("xyz", euler, degrees=False).as_quat()
    q_new = (Rotation.from_quat(q) * Rotation.from_rotvec(gyro_b * dt)).as_quat()
    euler_new = Rotation.from_quat(q_new).as_euler("xyz")
    R = Rotation.from_quat(q_new).as_matrix()
    a_w = R @ accel_b + g_world
    v_new = v + a_w * dt
    p_new = p + v * dt + 0.5 * a_w * (dt ** 2)
    return np.concatenate([p_new, v_new, euler_new])


def _measure_body_velocity(x: np.ndarray) -> np.ndarray:
    """观测模型：机体线速度 v_b = R^T * v_w。"""
    v = x[3:6]
    euler = x[6:9]
    R = Rotation.from_euler("xyz", euler, degrees=False).as_matrix()
    return R.T @ v


def _jacobian_predict(
    x: np.ndarray,
    accel_b: np.ndarray,
    gyro_b: np.ndarray,
    dt: float,
    g_world: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    x0 = _predict_state_ekf(x, accel_b, gyro_b, dt, g_world)
    F = np.zeros((9, 9), dtype=np.float64)
    for j in range(9):
        dx = np.zeros(9, dtype=np.float64)
        dx[j] = eps
        F[:, j] = (_predict_state_ekf(x + dx, accel_b, gyro_b, dt, g_world) - x0) / eps
    return F


def _jacobian_measure(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    h0 = _measure_body_velocity(x)
    H = np.zeros((3, 9), dtype=np.float64)
    for j in range(9):
        dx = np.zeros(9, dtype=np.float64)
        dx[j] = eps
        H[:, j] = (_measure_body_velocity(x + dx) - h0) / eps
    return H


def _smooth_wheel_velocity_body(v_body_meas: np.ndarray, ema_alpha: float) -> np.ndarray:
    """
    对机体轮速观测做指数平滑：s[i]=alpha*z[i]+(1-alpha)*s[i-1]，减轻高频锯齿。
    alpha 越小越平滑（建议 0.15~0.5）；alpha=1 等价于不平滑。
    """
    v = np.asarray(v_body_meas, dtype=np.float64)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    n = v.shape[0]
    out = np.empty_like(v)
    out[0] = v[0]
    a = float(ema_alpha)
    for i in range(1, n):
        out[i] = a * v[i] + (1.0 - a) * out[i - 1]
    return out.reshape(v_body_meas.shape)


class EkfImuWheelFilter:
    """
    增量式 IMU + 底盘轮速 EKF，与 ``fuse_pose_ekf`` 单步数学一致，供实时/流式调用。

    用法：在相邻 IMU 采样时刻 ``t_k -> t_{k+1}`` 调用 ``step``，传入 ``t_k`` 处的加计/陀螺
    与 ``t_{k+1}`` 处的轮速机体速度观测（与批处理中插值到 IMU 时刻的 ``v_body_meas`` 一致）。
    """

    def __init__(
        self,
        *,
        g_world: np.ndarray = G_WORLD_ENU,
        sigma_wheel: float = EKF_SIGMA_WHEEL,
        sigma_pos_process: float = EKF_SIGMA_POS_PROCESS,
        sigma_vel_process: float = EKF_SIGMA_VEL_PROCESS,
        sigma_euler_process: float = EKF_SIGMA_EULER_PROCESS,
        p_init_diag: Optional[np.ndarray] = None,
        wheel_update_every: int = 1,
    ) -> None:
        if int(wheel_update_every) < 1:
            raise ValueError("wheel_update_every 须为 >= 1 的整数")
        self._g_world = np.asarray(g_world, dtype=np.float64)
        self._sigma_wheel = float(sigma_wheel)
        self._sigma_pos_process = float(sigma_pos_process)
        self._sigma_vel_process = float(sigma_vel_process)
        self._sigma_euler_process = float(sigma_euler_process)
        self._stride = int(wheel_update_every)
        self._R_meas = np.eye(3, dtype=np.float64) * (self._sigma_wheel ** 2)
        self._I9 = np.eye(9, dtype=np.float64)

        if p_init_diag is None:
            p_init_diag = EKF_P_INIT_DIAG.copy()
        self.x = np.zeros(9, dtype=np.float64)
        self.P = np.diag(np.asarray(p_init_diag, dtype=np.float64) ** 2)
        self._predict_count = 0

    def pose_from_state(self) -> Tuple[np.ndarray, np.ndarray]:
        pos = self.x[0:3].copy()
        quat = Rotation.from_euler("xyz", self.x[6:9], degrees=False).as_quat()
        return pos, quat

    def step(
        self,
        accel_b: np.ndarray,
        gyro_b: np.ndarray,
        dt: float,
        v_body_meas_at_end: np.ndarray,
        *,
        force_wheel_update: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        推进到下一 IMU 时刻。``v_body_meas_at_end`` 为该时刻机体线速度观测（与批处理 ``z`` 一致）。
        ``force_wheel_update=True`` 时强制做一次轮速更新（批处理最后一 IMU 步）。
        """
        if dt <= 0.0:
            raise ValueError(f"dt 须 > 0，当前 dt={dt}")
        accel_b = np.asarray(accel_b, dtype=np.float64).ravel()
        gyro_b = np.asarray(gyro_b, dtype=np.float64).ravel()
        z = np.asarray(v_body_meas_at_end, dtype=np.float64).ravel()
        if z.shape[0] != 3:
            raise ValueError("v_body_meas_at_end 须为长度 3")

        sp = self._sigma_pos_process
        sv = self._sigma_vel_process
        se = self._sigma_euler_process
        g_world = self._g_world
        Q = np.diag(
            np.concatenate(
                [
                    np.full(3, (sp * dt) ** 2),
                    np.full(3, (sv * dt) ** 2),
                    np.full(3, (se * dt) ** 2),
                ]
            )
        )

        x_pred = _predict_state_ekf(self.x, accel_b, gyro_b, dt, g_world)
        F = _jacobian_predict(self.x, accel_b, gyro_b, dt, g_world)
        P_pred = F @ self.P @ F.T + Q

        self._predict_count += 1
        pc = self._predict_count
        st = self._stride
        do_wheel = force_wheel_update or (st == 1) or (pc % st == 0)

        if do_wheel:
            h = _measure_body_velocity(x_pred)
            H = _jacobian_measure(x_pred)
            S = H @ P_pred @ H.T + self._R_meas
            K = np.linalg.solve(S, (P_pred @ H.T).T).T
            y = z - h
            self.x = x_pred + K @ y
            self.P = (self._I9 - K @ H) @ P_pred
        else:
            self.x = x_pred
            self.P = P_pred

        return self.pose_from_state()


def fuse_pose_ekf(
    imu_data: np.ndarray,
    odom_data: np.ndarray,
    *,
    g_world: np.ndarray = G_WORLD_ENU,
    sigma_wheel: float = EKF_SIGMA_WHEEL,
    sigma_pos_process: float = EKF_SIGMA_POS_PROCESS,
    sigma_vel_process: float = EKF_SIGMA_VEL_PROCESS,
    sigma_euler_process: float = EKF_SIGMA_EULER_PROCESS,
    p_init_diag: Optional[np.ndarray] = None,
    wheel_meas_ema_alpha: float = 1.0,
    wheel_update_every: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    扩展卡尔曼滤波：IMU（加计+陀螺）预测，底盘机体线速度观测更新（底盘速度经 R_BASE_TO_IMU 对齐）。
    不使用 IMU 四元数列；初始姿态为单位阵。

    wheel_meas_ema_alpha：对轮速观测的指数平滑系数 (0,1]，越小越平滑；1 表示不平滑。

    wheel_update_every：每 N 次 **IMU 预测** 后做一次轮速观测更新（N=1 与原先「每步都更新」一致）。
    中间步只做预测、不融合轮速；最后一步总会做一次更新（若尚未更新）。

    返回 (t, position_w, quaternion_xyzw)，时间轴为重叠段内 IMU 时刻；
    起点位置为世界原点；输出四元数由滤波器欧拉角导出。
    """
    imu_data = ensure_2d(imu_data)
    odom_data = ensure_2d(odom_data)
    if imu_data.shape[1] < IMU_MIN_COLS:
        raise ValueError(f"IMU 至少需要 {IMU_MIN_COLS} 列（时间+加计+陀螺），当前 {imu_data.shape[1]}")
    if int(wheel_update_every) < 1:
        raise ValueError("wheel_update_every 须为 >= 1 的整数")
    stride = int(wheel_update_every)

    t_imu = imu_data[:, IMU_COL_TS]
    t_odom = odom_data[:, ODOM_COL_TS]
    t0 = max(float(t_imu.min()), float(t_odom.min()))
    t1 = min(float(t_imu.max()), float(t_odom.max()))
    if not (t0 < t1):
        raise ValueError(
            f"IMU 与底盘时间无有效重叠: imu [{t_imu.min():.6f},{t_imu.max():.6f}], "
            f"odom [{t_odom.min():.6f},{t_odom.max():.6f}]"
        )

    mask = (t_imu >= t0) & (t_imu <= t1)
    if not np.any(mask):
        raise ValueError("重叠区间内无 IMU 样本")

    imu_s = imu_data[mask]
    t = imu_s[:, IMU_COL_TS]
    n = len(t)
    if n < 2:
        q_id = _quat_identity_xyzw().reshape(1, 4)
        return t, np.zeros((1, 3), dtype=np.float64), align_quaternion_continuous(q_id)

    v_base_meas = _interp_linear_xyz(t, t_odom, odom_data[:, ODOM_COL_LXYZ])
    v_body_meas = base_velocity_to_imu_body(v_base_meas)
    ema = float(wheel_meas_ema_alpha)
    if 0.0 < ema < 1.0:
        v_body_meas = _smooth_wheel_velocity_body(v_body_meas, ema)

    flt = EkfImuWheelFilter(
        g_world=g_world,
        sigma_wheel=sigma_wheel,
        sigma_pos_process=sigma_pos_process,
        sigma_vel_process=sigma_vel_process,
        sigma_euler_process=sigma_euler_process,
        p_init_diag=p_init_diag,
        wheel_update_every=wheel_update_every,
    )

    pos_out = np.zeros((n, 3), dtype=np.float64)
    quat_out = np.zeros((n, 4), dtype=np.float64)
    quat_out[0] = _quat_identity_xyzw()
    pos_out[0] = np.zeros(3, dtype=np.float64)

    for i in range(n - 1):
        dt = float(t[i + 1] - t[i])
        if dt <= 0.0:
            raise ValueError(f"时间须严格递增，在索引 {i} 处 dt={dt}")
        accel_b = imu_s[i, IMU_COL_ACC]
        gyro_b = imu_s[i, IMU_COL_GYRO]
        force_last = i == n - 2
        p_i, q_i = flt.step(
            accel_b,
            gyro_b,
            dt,
            v_body_meas[i + 1],
            force_wheel_update=force_last,
        )
        pos_out[i + 1] = p_i
        quat_out[i + 1] = q_i

    quat_out = align_quaternion_continuous(quat_out)
    return t, pos_out, quat_out
