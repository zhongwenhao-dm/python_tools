import argparse
import os
import sys
from pathlib import Path

# 允许从任意工作目录执行 `python .../imu_lekiwi_fusion.py`，同目录模块可被导入
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np

from planar_chassis_dr import compute_chassis_dr_for_fusion_grid
from pose_fusion import (
    IMU_MIN_COLS,
    EKF_SIGMA_WHEEL,
    ensure_2d,
    fuse_pose_ekf,
    fuse_pose_imu_odom,
)


def load_csv_numeric(path: str) -> np.ndarray:
    """Comma-separated floats; line 1 is header, data starts at line 2."""
    return np.loadtxt(path, delimiter=",", dtype=np.float64, skiprows=1)


def save_pose_csv(path: str, t: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> None:
    header = "timestamp,x,y,z,qx,qy,qz,qw"
    data = np.column_stack([t, pos, quat])
    np.savetxt(path, data, delimiter=",", header=header, comments="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMU + 底盘 → 世界系位姿（链式或 EKF）")
    parser.add_argument("--data_dir", default="/home/dmgz/ZWH/lerobot/data/lekiwi/my_awesome_kiwi_20260403_161202", type=str)
    parser.add_argument("--imu_file_name", default="imu.txt", type=str)
    parser.add_argument("--odom_file_name", default="lekiwi_base_velocity.txt", type=str)
    parser.add_argument(
        "--fusion",
        choices=("ekf", "chain"),
        default="ekf",
        help="ekf：EKF（加计+陀螺预测+轮速观测）；chain：陀螺积分姿态+底盘速度链式积分（均不用 IMU 四元数列）",
    )
    parser.add_argument(
        "--out",
        default="",
        type=str,
        help="输出轨迹 CSV（默认: data_dir/fused_pose.csv）",
    )
    parser.add_argument("--vis", action="store_true", help="融合完成后显示位姿轨迹图")
    parser.add_argument(
        "--save_fig",
        default="",
        type=str,
        help="将位姿图保存为 PNG（可与 --vis 同用；仅保存时可不写 --vis）",
    )
    parser.add_argument(
        "--compare_dr",
        action="store_true",
        help="Plot chassis planar DR vs fused position (same time grid)",
    )
    parser.add_argument(
        "--save_compare",
        default="",
        type=str,
        help="Save DR vs fusion comparison PNG (default: data_dir/fusion_vs_dr_compare.png)",
    )
    parser.add_argument(
        "--sigma_wheel",
        type=float,
        default=None,
        help=f"EKF wheel velocity meas. noise σ (m/s); default {EKF_SIGMA_WHEEL} from pose_fusion (larger → smoother)",
    )
    parser.add_argument(
        "--wheel_ema",
        type=float,
        default=1.0,
        help="EKF: EMA on wheel v_body before update, alpha in (0,1]; 1=no smoothing, ~0.2–0.4 smoother",
    )
    parser.add_argument(
        "--wheel_update_every",
        type=int,
        default=1,
        help="EKF: 每 N 次 IMU 预测后做一次轮速更新（1=每步更新；例如 5 表示约每 5 个 IMU 步才用轮速修正一次）",
    )
    args = parser.parse_args()

    imu_data = load_csv_numeric(os.path.join(args.data_dir, args.imu_file_name))
    odom_data = load_csv_numeric(os.path.join(args.data_dir, args.odom_file_name))

    imu_data = ensure_2d(imu_data)
    odom_data = ensure_2d(odom_data)

    print("imu:", imu_data.shape, "odom:", odom_data.shape, "fusion:", args.fusion)
    if imu_data.shape[1] < IMU_MIN_COLS:
        print(f"警告: imu 至少需要 {IMU_MIN_COLS} 列 (t+acc+gyro)，当前列数:", imu_data.shape[1])
    if odom_data.shape[1] != 7:
        print("警告: odom 期望 7 列 (timestamp + 6)，当前列数:", odom_data.shape[1])

    if args.fusion == "ekf":
        ekf_kw = {}
        if args.sigma_wheel is not None:
            ekf_kw["sigma_wheel"] = args.sigma_wheel
        if args.wheel_ema < 1.0:
            if not (0.0 < args.wheel_ema <= 1.0):
                print("警告: --wheel_ema 应在 (0,1]，已忽略", file=sys.stderr)
            else:
                ekf_kw["wheel_meas_ema_alpha"] = args.wheel_ema
        if args.wheel_update_every != 1:
            ekf_kw["wheel_update_every"] = args.wheel_update_every
        t, pos, quat = fuse_pose_ekf(imu_data, odom_data, **ekf_kw)
    else:
        t, pos, quat = fuse_pose_imu_odom(imu_data, odom_data)

    out_path = args.out or os.path.join(args.data_dir, "fused_pose.csv")
    save_pose_csv(out_path, t, pos, quat)
    print(f"已写入 {len(t)} 帧 -> {out_path}")
    print("末位置 [x,y,z]:", pos[-1])

    fusion_label = "EKF" if args.fusion == "ekf" else "chain"

    if args.vis or args.save_fig:
        from pose_visualization import visualize_pose_trajectory

        fig_path = args.save_fig or None
        visualize_pose_trajectory(
            t,
            pos,
            quat,
            title=f"Fused pose ({fusion_label})",
            save_path=fig_path,
            show=bool(args.vis),
        )
        if fig_path:
            print(f"位姿图已保存 -> {fig_path}")

    if args.compare_dr:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("跳过对比图: 未安装 matplotlib", file=sys.stderr)
        else:
            if len(t) < 2:
                print("跳过对比图: 融合样本少于 2")
            else:
                try:
                    pos_ch = compute_chassis_dr_for_fusion_grid(odom_data, t)
                except ValueError as e:
                    print(f"跳过对比图: {e}")
                else:
                    from pose_visualization import plot_fusion_vs_dr

                    cmp_fig = plot_fusion_vs_dr(pos_ch, pos, fusion_name=fusion_label)
                    cmp_out = args.save_compare or os.path.join(args.data_dir, "fusion_vs_dr_compare.png")
                    parent = os.path.dirname(os.path.abspath(cmp_out))
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    cmp_fig.savefig(cmp_out, dpi=150, bbox_inches="tight")
                    print(f"DR vs fusion 对比图已保存 -> {cmp_out}")
                    if args.vis:
                        plt.show()
                    else:
                        plt.close(cmp_fig)
