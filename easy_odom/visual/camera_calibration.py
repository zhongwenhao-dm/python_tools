"""
LeKiwi 相机标定：PINHOLE 内参 + 相机系 → IMU 系外参。

约定：相机坐标系下齐次点 ``p_c`` 变到 IMU 系为 ``p_i = T_ic @ p_c``（4×4）。

本文件仅集中保存标定数值，供其它模块或项目 ``import`` 使用；easy_odom 内无视觉/融合逻辑。
修改标定请只编辑本文件。
"""
from __future__ import annotations

import numpy as np

CAMERA_MODEL = "PINHOLE"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

CAMERA_FX = 759.68
CAMERA_FY = 757.44
CAMERA_CX = 302.09
CAMERA_CY = 200.55

# OpenCV 畸变：k1, k2, p1, p2（与常见 plumb-bob 四系数一致）
CAMERA_DISTORTION = np.array(
    [-0.016744, -0.115349, 0.009929, -0.007112],
    dtype=np.float64,
)

T_IC = np.array(
    [
        [1.0, 0.0, 0.0, 0.70],
        [0.0, 0.0, 1.0, 0.40],
        [0.0, -1.0, 0.0, 0.40],
        [0.0, 0.0, 0.0, 1.00],
    ],
    dtype=np.float64,
)

# 相机系 → IMU 系旋转（平移仅用于点变换；方向向量只用 R）
R_CAM_TO_IMU = T_IC[:3, :3].copy()


def pinhole_K() -> np.ndarray:
    """标称分辨率 640×480 下的 3×3 相机矩阵。"""
    return np.array(
        [
            [CAMERA_FX, 0.0, CAMERA_CX],
            [0.0, CAMERA_FY, CAMERA_CY],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def pinhole_K_for_image_size(height: int, width: int) -> np.ndarray:
    """
    若实际图像尺寸与标定分辨率不一致，按像素比例缩放 fx,fy,cx,cy。
    此时畸变不再与原始标定一致，调用方可对畸变采用近似（如不同分辨率时置零）。
    """
    K = pinhole_K()
    if width == CAMERA_WIDTH and height == CAMERA_HEIGHT:
        return K
    sx = float(width) / float(CAMERA_WIDTH)
    sy = float(height) / float(CAMERA_HEIGHT)
    return np.array(
        [
            [K[0, 0] * sx, 0.0, K[0, 2] * sx],
            [0.0, K[1, 1] * sy, K[1, 2] * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def distortion_for_image_size(height: int, width: int) -> np.ndarray:
    """仅在与标定分辨率一致时使用原始畸变；否则返回零（近似）。"""
    if width == CAMERA_WIDTH and height == CAMERA_HEIGHT:
        return CAMERA_DISTORTION.copy()
    return np.zeros(4, dtype=np.float64)


__all__ = [
    "CAMERA_MODEL",
    "CAMERA_WIDTH",
    "CAMERA_HEIGHT",
    "CAMERA_DISTORTION",
    "T_IC",
    "R_CAM_TO_IMU",
    "pinhole_K",
    "pinhole_K_for_image_size",
    "distortion_for_image_size",
]
