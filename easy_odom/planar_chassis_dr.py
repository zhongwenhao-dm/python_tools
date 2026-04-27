"""Chassis-only planar dead reckoning helpers (used by imu_lekiwi_fusion compare)."""
from __future__ import annotations

import numpy as np

from pose_fusion import (
    ODOM_COL_LXYZ,
    ODOM_COL_TS,
    base_velocity_to_imu_body,
    ensure_2d,
    interp_linear_1d,
)
from pose_fusion import _interp_linear_xyz as interp_linear_xyz  # noqa: SLF001

# Column 6 = yaw rate deg/s (0-based; column 0 is timestamp).
CHASSIS_YAW_COL = 6
DEG2RAD = np.pi / 180.0


def integrate_chassis_planar_dr(
    t: np.ndarray,
    v_base: np.ndarray,
    yaw_rate_deg_s: np.ndarray,
) -> np.ndarray:
    """Yaw rate in deg/s; [lx,ly,lz] via R_BASE_TO_IMU."""
    n = len(t)
    pos = np.zeros((n, 3), dtype=np.float64)
    if n < 2:
        return pos

    yaw = 0.0
    for i in range(n - 1):
        dt = float(t[i + 1] - t[i])
        if dt <= 0:
            raise ValueError(f"Timestamps must be strictly increasing (dt={dt} at index {i})")

        c, s = np.cos(yaw), np.sin(yaw)
        v3 = base_velocity_to_imu_body(v_base[i])
        vbx, vby = float(v3[0]), float(v3[1])
        v_wx = c * vbx - s * vby
        v_wy = s * vbx + c * vby
        pos[i + 1, 0] = pos[i, 0] + v_wx * dt
        pos[i + 1, 1] = pos[i, 1] + v_wy * dt

        omega_rad_s = float(yaw_rate_deg_s[i]) * DEG2RAD
        yaw += omega_rad_s * dt

    return pos


def compute_chassis_dr_for_fusion_grid(odom_data: np.ndarray, t_fused: np.ndarray) -> np.ndarray:
    """Chassis planar DR on the same time stamps as fusion (odom interpolated to t_fused)."""
    odom_data = ensure_2d(odom_data)
    t_odom = odom_data[:, ODOM_COL_TS]
    v_base_t = interp_linear_xyz(t_fused, t_odom, odom_data[:, ODOM_COL_LXYZ])
    yaw_chassis_t = interp_linear_1d(t_fused, t_odom, odom_data[:, CHASSIS_YAW_COL])
    return integrate_chassis_planar_dr(t_fused, v_base_t, yaw_chassis_t)


__all__ = [
    "CHASSIS_YAW_COL",
    "DEG2RAD",
    "integrate_chassis_planar_dr",
    "compute_chassis_dr_for_fusion_grid",
]
