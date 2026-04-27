"""
融合位姿轨迹可视化：3D 轨迹、XY 俯视图、稀疏机体坐标轴；
以及底盘平面 DR 与融合轨迹对比图 plot_fusion_vs_dr。
依赖：numpy, scipy, matplotlib。
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except ImportError as e:  # pragma: no cover
    raise ImportError("位姿可视化需要 matplotlib，请安装: pip install matplotlib") from e


def load_fused_pose_csv(path: str, skip_header: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取 fused_pose.csv：timestamp,x,y,z,qx,qy,qz,qw
    返回 (t, pos, quat_xyzw)。
    """
    kw: dict = {"delimiter": ",", "dtype": np.float64}
    if skip_header:
        kw["skiprows"] = 1
    data = np.loadtxt(path, **kw)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 8:
        raise ValueError(f"期望 8 列，得到 {data.shape[1]} 列: {path}")
    t = data[:, 0]
    pos = data[:, 1:4]
    quat = data[:, 4:8]
    return t, pos, quat


def _xy_equal_limits_union(ax, pos_list: Tuple[np.ndarray, ...]) -> None:
    """XY 等比例；范围取多条轨迹在平面上的并集。"""
    xy = np.vstack([p[:, :2] for p in pos_list])
    xmin, ymin = float(np.min(xy[:, 0])), float(np.min(xy[:, 1]))
    xmax, ymax = float(np.max(xy[:, 0])), float(np.max(xy[:, 1]))
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    span = max(xmax - xmin, ymax - ymin, 1e-9)
    r = span / 2.0
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_aspect("equal", adjustable="box")


def _apply_3d_equal_scale(ax, pos: np.ndarray, margin: float) -> None:
    """使 3D 子图 x/y/z 使用相同米制比例（立方体视窗），避免某一轴被压扁。"""
    mins = np.min(pos, axis=0)
    maxs = np.max(pos, axis=0)
    cen = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins))
    if span < 1e-9:
        span = 1.0
    r = span / 2.0 + float(margin)
    ax.set_xlim(cen[0] - r, cen[0] + r)
    ax.set_ylim(cen[1] - r, cen[1] + r)
    ax.set_zlim(cen[2] - r, cen[2] + r)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))


def plot_fusion_vs_dr(
    pos_ch: np.ndarray,
    pos_fused: np.ndarray,
    *,
    fusion_name: str = "EKF",
) -> Figure:
    """
    Chassis planar DR vs fused position: XY + 3D (English labels, equal scales).
    """
    pos_ch = np.asarray(pos_ch, dtype=np.float64)
    pos_fused = np.asarray(pos_fused, dtype=np.float64)
    fig = plt.figure(figsize=(12, 5))
    ax_xy = fig.add_subplot(121)
    ax_xy.plot(pos_ch[:, 0], pos_ch[:, 1], color="C0", lw=1.2, label="chassis DR")
    ax_xy.plot(pos_fused[:, 0], pos_fused[:, 1], color="C1", lw=1.2, label="fused")
    ax_xy.scatter(pos_ch[0, 0], pos_ch[0, 1], c="green", s=36, zorder=5, label="start")
    _xy_equal_limits_union(ax_xy, (pos_ch, pos_fused))
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("XY (chassis DR vs fused)")
    ax_xy.grid(True, alpha=0.35)
    ax_xy.legend(loc="best", fontsize=8)

    pos_all = np.vstack([pos_ch, pos_fused])
    ax3 = fig.add_subplot(122, projection="3d")
    ax3.plot(pos_ch[:, 0], pos_ch[:, 1], pos_ch[:, 2], color="C0", lw=1.0, label="chassis DR")
    ax3.plot(pos_fused[:, 0], pos_fused[:, 1], pos_fused[:, 2], color="C1", lw=1.0, label="fused")
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_zlabel("Z (m)")
    ax3.set_title("3D")
    ax3.legend(loc="upper right", fontsize=8)
    _apply_3d_equal_scale(ax3, pos_all, margin=0.05)

    fig.suptitle(f"Chassis DR vs fused ({fusion_name})", fontsize=11)
    fig.tight_layout()
    return fig


def visualize_pose_trajectory(
    t: np.ndarray,
    pos: np.ndarray,
    quat: np.ndarray,
    *,
    axis_step: int = 0,
    axis_length: float = 0.08,
    title: str = "Fused pose",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    绘制 3D 轨迹 + XY 俯视图；每隔 axis_step 帧绘制机体 x/y/z 轴（0 表示自动约 50 根箭头）。
    3D 子图中 x/y/z 轴采用相同数据尺度（等比例立方体视窗）。

    quat: (N,4) x,y,z,w，与 scipy Rotation.from_quat 一致。
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    pos = np.asarray(pos, dtype=np.float64)
    quat = np.asarray(quat, dtype=np.float64)
    if pos.ndim == 1:
        pos = pos.reshape(1, -1)
    n = len(t)
    if len(pos) != n or len(quat) != n:
        raise ValueError("t、pos、quat 行数须一致")

    if axis_step <= 0:
        axis_step = max(1, n // 50)

    fig = plt.figure(figsize=(12.5, 5.2))
    ax3d = fig.add_subplot(121, projection="3d")
    axy = fig.add_subplot(122)

    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], color="steelblue", linewidth=1.2, label="path")
    ax3d.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color="green", s=40, label="start", zorder=5)
    ax3d.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], color="crimson", s=40, label="end", zorder=5)

    rot = Rotation.from_quat(quat)
    idx = np.arange(0, n, axis_step, dtype=int)
    L = float(axis_length)
    for k in idx:
        R = rot[k].as_matrix()
        p = pos[k]
        # 机体系 x/y/z 在世界系中的方向为 R 的列
        for col, color in ((0, "tab:red"), (1, "tab:green"), (2, "tab:blue")):
            d = R[:, col]
            ax3d.quiver(
                p[0],
                p[1],
                p[2],
                d[0],
                d[1],
                d[2],
                color=color,
                length=L,
                normalize=True,
                linewidth=0.8,
                alpha=0.85,
            )

    span_data = float(np.max(np.max(pos, axis=0) - np.min(pos, axis=0)))
    margin = max(L * 1.25, 0.02 * span_data if span_data > 1e-9 else 0.05)
    _apply_3d_equal_scale(ax3d, pos, margin=margin)

    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3D trajectory")
    ax3d.legend(loc="upper right", fontsize=8)

    axy.plot(pos[:, 0], pos[:, 1], color="steelblue", lw=1.2)
    axy.scatter(pos[0, 0], pos[0, 1], c="green", s=35, zorder=5, label="start")
    axy.scatter(pos[-1, 0], pos[-1, 1], c="crimson", s=35, zorder=5, label="end")
    axy.set_xlabel("X (m)")
    axy.set_ylabel("Y (m)")
    axy.set_title("XY top view")
    axy.set_aspect("equal", adjustable="box")
    axy.grid(True, alpha=0.35)
    axy.legend(loc="best", fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if save_path:
        parent = os.path.dirname(os.path.abspath(save_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化 fused_pose.csv 位姿轨迹")
    parser.add_argument("pose_csv", type=str, help="fused_pose.csv 路径")
    parser.add_argument("--no_header", action="store_true", help="CSV 无表头")
    parser.add_argument("--out", default="", type=str, help="保存 PNG 路径（可选）")
    parser.add_argument("--no-show", action="store_true", help="不弹窗，仅保存")
    args = parser.parse_args()

    tt, pp, qq = load_fused_pose_csv(args.pose_csv, skip_header=not args.no_header)
    out = args.out or ""
    visualize_pose_trajectory(
        tt,
        pp,
        qq,
        title=os.path.basename(args.pose_csv),
        save_path=out if out else None,
        show=not args.no_show,
    )
