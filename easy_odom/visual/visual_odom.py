#!/usr/bin/env python3
"""
Monocular visual odometry for LeKiwi camera images.

默认读取数据目录：
- camera_image_compressed/*.png   文件名形如 1775203928_253302915.png
- lekiwi_base_velocity.txt        可选，用于给单目 VO 恢复近似米制尺度

算法：
- ORB 特征 + Hamming KNN ratio matching
- Essential Matrix + recoverPose 得到相邻帧相对旋转/平移方向
- 单目尺度默认从 lekiwi_base_velocity.txt 的线速度模长积分得到
- 实时显示前后帧追踪匹配和当前轨迹
- 输出 timestamp,x,y,z,qx,qy,qz,qw

注意：这是轻量基线 VO。单目视觉对光照、纹理、动态物体和尺度都很敏感；若轮速尺度不可用，
输出轨迹只表示相对形状，不是米制。
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise SystemExit("需要安装 opencv-python: pip install opencv-python") from e

_VISUAL_DIR = Path(__file__).resolve().parent
_EASY_ODOM_ROOT = _VISUAL_DIR.parent
if str(_VISUAL_DIR) not in sys.path:
    sys.path.insert(0, str(_VISUAL_DIR))
if str(_EASY_ODOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_EASY_ODOM_ROOT))

from camera_calibration import distortion_for_image_size, pinhole_K_for_image_size  # noqa: E402


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

IMAGE_DIR_NAME = "camera_image_compressed"
ODOM_FILE_NAME = "lekiwi_base_velocity.txt"
DEFAULT_OUT_NAME = "visual_odom_pose.csv"

ORB_N_FEATURES = 2500
ORB_FAST_THRESHOLD = 12
MATCH_RATIO = 0.75
MIN_MATCHES = 80
MIN_INLIERS = 50
ESSENTIAL_RANSAC_PROB = 0.999
ESSENTIAL_RANSAC_THRESH_PX = 1.5
MAX_FRAME_STEP = 1
DEFAULT_IMAGE_STEP = 5
REALTIME_MAX_MATCH_DRAW = 80
REALTIME_PAUSE_SEC = 0.001

# recoverPose 的平移方向偶尔会跳；如果某步轮速尺度很小，则直接只积分旋转，避免噪声位移。
MIN_SCALE_M = 1e-4

_IMAGE_TS_RE = re.compile(r"^(\d+)_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)


@dataclass(frozen=True)
class ImageFrame:
    path: str
    timestamp: float


@dataclass
class VoStats:
    total_pairs: int = 0
    used_pairs: int = 0
    skipped_matches: int = 0
    skipped_essential: int = 0
    skipped_inliers: int = 0


@dataclass(frozen=True)
class RelativePoseResult:
    rotation_21: Rotation
    translation_21_dir: np.ndarray
    inlier_count: int
    inlier_matches: list[cv2.DMatch]


def parse_image_timestamp(path: str) -> float | None:
    """从 1775203928_253302915.png 解析 Unix 秒时间戳。"""
    name = os.path.basename(path)
    m = _IMAGE_TS_RE.match(name)
    if not m:
        return None
    sec = int(m.group(1))
    nsec = int(m.group(2).ljust(9, "0")[:9])
    return float(sec) + float(nsec) * 1e-9


def list_image_frames(
    image_dir: str,
    max_frames: int = 0,
    step: int = MAX_FRAME_STEP,
    start_ratio: float = 0.0,
    end_ratio: float = 100.0,
) -> list[ImageFrame]:
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")
    frames: list[ImageFrame] = []
    for name in os.listdir(image_dir):
        path = os.path.join(image_dir, name)
        if not os.path.isfile(path):
            continue
        ts = parse_image_timestamp(path)
        if ts is None:
            continue
        frames.append(ImageFrame(path=path, timestamp=ts))
    frames.sort(key=lambda f: f.timestamp)
    if not (0.0 <= start_ratio < end_ratio <= 100.0):
        raise ValueError(
            f"ratio 范围须满足 0 <= start < end <= 100，当前 {start_ratio}, {end_ratio}"
        )
    n_all = len(frames)
    start_idx = int(np.floor(n_all * (float(start_ratio) / 100.0)))
    end_idx = int(np.ceil(n_all * (float(end_ratio) / 100.0)))
    frames = frames[start_idx:end_idx]
    if step > 1:
        frames = frames[:: int(step)]
    if max_frames > 0:
        frames = frames[: int(max_frames)]
    if len(frames) < 2:
        raise ValueError(f"图片帧不足: {image_dir}, parsed={len(frames)}")
    return frames


def load_odom_velocity(path: str) -> Tuple[np.ndarray, np.ndarray] | None:
    """
    读取 lekiwi_base_velocity.txt，返回 (t, speed_magnitude)。
    文件格式：timestamp,lx,ly,lz,ax,ay,az。
    """
    if not os.path.isfile(path):
        return None
    data = np.loadtxt(path, delimiter=",", dtype=np.float64, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"底盘速度文件至少需要 4 列: {path}")
    t = data[:, 0]
    speed = np.linalg.norm(data[:, 1:4], axis=1)
    order = np.argsort(t)
    return t[order], speed[order]


def integrate_speed_between(t_src: np.ndarray, speed: np.ndarray, t0: float, t1: float) -> float:
    """线性插值速度并梯形积分，估计 [t0,t1] 位移尺度。"""
    if t1 <= t0:
        return 0.0
    lo = max(float(t_src[0]), float(t0))
    hi = min(float(t_src[-1]), float(t1))
    if hi <= lo:
        return 0.0
    mask = (t_src > lo) & (t_src < hi)
    ts = np.concatenate(([lo], t_src[mask], [hi]))
    vs = np.interp(ts, t_src, speed)
    return float(np.trapz(vs, ts))


def image_to_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图片: {path}")
    return img


def undistort_if_needed(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape[:2]
    k = pinhole_K_for_image_size(h, w)
    dist = distortion_for_image_size(h, w)
    if np.linalg.norm(dist) > 0.0:
        gray = cv2.undistort(gray, k, dist)
    return gray, k


def detect_and_compute(orb: cv2.ORB, gray: np.ndarray):
    keypoints, desc = orb.detectAndCompute(gray, None)
    if desc is None:
        desc = np.empty((0, 32), dtype=np.uint8)
    return keypoints, desc


def match_descriptors(desc0: np.ndarray, desc1: np.ndarray) -> list[cv2.DMatch]:
    if len(desc0) < 2 or len(desc1) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(desc0, desc1, k=2)
    good: list[cv2.DMatch] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < MATCH_RATIO * n.distance:
            good.append(m)
    return good


def relative_pose_from_matches(
    kp0: Iterable[cv2.KeyPoint],
    kp1: Iterable[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    k: np.ndarray,
) -> RelativePoseResult | None:
    if len(matches) < MIN_MATCHES:
        return None
    kp0_list = list(kp0)
    kp1_list = list(kp1)
    pts0 = np.float32([kp0_list[m.queryIdx].pt for m in matches])
    pts1 = np.float32([kp1_list[m.trainIdx].pt for m in matches])
    essential, mask = cv2.findEssentialMat(
        pts0,
        pts1,
        k,
        method=cv2.RANSAC,
        prob=ESSENTIAL_RANSAC_PROB,
        threshold=ESSENTIAL_RANSAC_THRESH_PX,
    )
    if essential is None or mask is None:
        return None
    if essential.shape[0] > 3:
        essential = essential[:3, :]
    inlier_count = int(np.count_nonzero(mask))
    if inlier_count < MIN_INLIERS:
        return None
    recover_inliers, r_mat, t_vec, pose_mask = cv2.recoverPose(
        essential,
        pts0,
        pts1,
        k,
        mask=mask,
    )
    if int(recover_inliers) < MIN_INLIERS:
        return None
    t = np.asarray(t_vec, dtype=np.float64).reshape(3)
    nrm = float(np.linalg.norm(t))
    if nrm < 1e-12:
        return None
    if pose_mask is not None:
        inlier_bool = pose_mask.ravel().astype(bool)
    else:
        inlier_bool = mask.ravel().astype(bool)
    inlier_matches = [m for m, keep in zip(matches, inlier_bool) if keep]
    return RelativePoseResult(
        rotation_21=Rotation.from_matrix(r_mat),
        translation_21_dir=t / nrm,
        inlier_count=int(recover_inliers),
        inlier_matches=inlier_matches,
    )


def draw_tracking_image(
    gray_prev: np.ndarray,
    kp_prev: list[cv2.KeyPoint],
    gray_cur: np.ndarray,
    kp_cur: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    max_draw: int = REALTIME_MAX_MATCH_DRAW,
) -> np.ndarray:
    """返回 RGB 图，用于 Matplotlib 展示前后帧匹配。"""
    if not matches:
        left = cv2.cvtColor(gray_prev, cv2.COLOR_GRAY2BGR)
        right = cv2.cvtColor(gray_cur, cv2.COLOR_GRAY2BGR)
        vis = np.hstack([left, right])
        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    matches_draw = sorted(matches, key=lambda m: m.distance)[: int(max_draw)]
    vis = cv2.drawMatches(
        gray_prev,
        kp_prev,
        gray_cur,
        kp_cur,
        matches_draw,
        None,
        matchColor=(0, 220, 0),
        singlePointColor=(120, 120, 120),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def run_visual_odometry_realtime(
    frames: list[ImageFrame],
    odom_velocity: tuple[np.ndarray, np.ndarray] | None,
    scale_mode: str,
    pause_sec: float = REALTIME_PAUSE_SEC,
    max_match_draw: int = REALTIME_MAX_MATCH_DRAW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, VoStats]:
    """运行 VO，同时实时显示匹配追踪和累计轨迹。"""
    plt = import_pyplot_without_qt(show=True)
    plt.ion()

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    ax_img, ax_xy = axes
    ax_img.set_axis_off()
    ax_xy.set_title("Realtime VO trajectory")
    ax_xy.set_xlabel("x (m if wheel-scaled)")
    ax_xy.set_ylabel("y (m if wheel-scaled)")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, alpha=0.35)

    orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES, fastThreshold=ORB_FAST_THRESHOLD)
    stats = VoStats(total_pairs=max(0, len(frames) - 1))

    prev_gray, k = undistort_if_needed(image_to_gray(frames[0].path))
    kp_prev, desc_prev = detect_and_compute(orb, prev_gray)
    prev_ts = frames[0].timestamp
    r_wc = Rotation.identity()
    p_w = np.zeros(3, dtype=np.float64)

    out_t = [prev_ts]
    out_pos = [p_w.copy()]
    out_quat = [r_wc.as_quat()]

    img_artist = ax_img.imshow(cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2RGB))
    (traj_line,) = ax_xy.plot([0.0], [0.0], color="C0", lw=1.3, label="VO")
    start_scatter = ax_xy.scatter([0.0], [0.0], c="green", s=42, label="start", zorder=5)
    cur_scatter = ax_xy.scatter([0.0], [0.0], c="crimson", s=42, label="current", zorder=5)
    ax_xy.legend(loc="best", fontsize=8)
    fig.tight_layout()
    plt.pause(max(float(pause_sec), 1e-6))

    for idx, frame in enumerate(frames[1:], start=1):
        gray, k_cur = undistort_if_needed(image_to_gray(frame.path))
        k = k_cur
        kp_cur, desc_cur = detect_and_compute(orb, gray)
        matches = match_descriptors(desc_prev, desc_cur)
        draw_matches = matches
        status = ""

        if len(matches) < MIN_MATCHES:
            stats.skipped_matches += 1
            status = f"frame {idx}/{len(frames)-1}: matches={len(matches)} < {MIN_MATCHES}"
        else:
            rel = relative_pose_from_matches(kp_prev, kp_cur, matches, k)
            if rel is None:
                stats.skipped_essential += 1
                status = f"frame {idx}/{len(frames)-1}: E failed, matches={len(matches)}"
            else:
                draw_matches = rel.inlier_matches or matches
                if scale_mode == "wheel" and odom_velocity is not None:
                    scale = integrate_speed_between(
                        odom_velocity[0], odom_velocity[1], prev_ts, frame.timestamp
                    )
                else:
                    scale = 1.0
                if scale >= MIN_SCALE_M:
                    delta_c1 = -rel.rotation_21.inv().apply(rel.translation_21_dir) * float(scale)
                    p_w = p_w + r_wc.apply(delta_c1)
                r_wc = r_wc * rel.rotation_21.inv()
                stats.used_pairs += 1
                status = (
                    f"frame {idx}/{len(frames)-1}: used, matches={len(matches)}, "
                    f"inliers={rel.inlier_count}, scale={scale:.4f}"
                )

        out_t.append(frame.timestamp)
        out_pos.append(p_w.copy())
        out_quat.append(r_wc.as_quat())

        track_rgb = draw_tracking_image(
            prev_gray,
            kp_prev,
            gray,
            kp_cur,
            draw_matches,
            max_draw=max_match_draw,
        )
        img_artist.set_data(track_rgb)
        ax_img.set_title(status, fontsize=9)

        pos_arr = np.asarray(out_pos)
        traj_line.set_data(pos_arr[:, 0], pos_arr[:, 1])
        cur_scatter.set_offsets([[p_w[0], p_w[1]]])
        start_scatter.set_offsets([[pos_arr[0, 0], pos_arr[0, 1]]])
        ax_xy.relim()
        ax_xy.autoscale_view()
        ax_xy.set_aspect("equal", adjustable="box")
        fig.canvas.draw_idle()
        plt.pause(max(float(pause_sec), 1e-6))
        if not plt.fignum_exists(fig.number):
            break

        kp_prev, desc_prev, prev_gray, prev_ts = kp_cur, desc_cur, gray, frame.timestamp

    plt.ioff()
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


def import_pyplot_without_qt(show: bool):
    """
    opencv-python 会把 Qt 插件路径指向 cv2/qt/plugins，Matplotlib 若选 Qt 后端可能触发 xcb 崩溃。
    这里显式避开 Qt：需要弹窗时优先 TkAgg；仅保存时使用 Agg。
    """
    try:
        import matplotlib

        if show:
            try:
                matplotlib.use("TkAgg", force=True)
            except Exception:
                matplotlib.use("Agg", force=True)
                print(
                    "警告: TkAgg 后端不可用，已切换 Agg；本次实时窗口可能无法弹出。",
                    file=sys.stderr,
                )
        else:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise SystemExit("需要 matplotlib 才能画图: pip install matplotlib") from e
    return plt


def main() -> None:
    parser = argparse.ArgumentParser(description="LeKiwi 单目视觉里程计（ORB + Essential + recoverPose）")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="/home/dmgz/ZWH/lerobot/data/lekiwi/my_awesome_kiwi_20260403_161202",
        help="数据目录，默认使用 my_awesome_kiwi_20260403_161202",
    )
    parser.add_argument("--image_dir", default="", help="图片目录；默认 data_dir/camera_image_compressed")
    parser.add_argument("--out", default="", help="输出 CSV；默认 data_dir/visual_odom_pose.csv")
    parser.add_argument(
        "--scale_mode",
        choices=("wheel", "unit"),
        default="wheel",
        help="wheel: 用 lekiwi_base_velocity.txt 恢复尺度；unit: 每步单位尺度",
    )
    parser.add_argument("--max_frames", type=int, default=0, help="最多处理多少帧；0 表示百分比裁剪后的全部")
    parser.add_argument("--start_ratio", type=float, default=0.0, help="从数据集百分之多少开始，默认 0")
    parser.add_argument("--end_ratio", type=float, default=100.0, help="到数据集百分之多少结束，默认 100")
    parser.add_argument(
        "--step",
        type=int,
        default=DEFAULT_IMAGE_STEP,
        help=f"图片抽帧步长；默认 {DEFAULT_IMAGE_STEP}，避免相邻帧基线过小导致 Essential 退化",
    )
    parser.add_argument(
        "--realtime_pause",
        type=float,
        default=REALTIME_PAUSE_SEC,
        help=f"实时显示每帧刷新暂停秒数，默认 {REALTIME_PAUSE_SEC}",
    )
    parser.add_argument(
        "--realtime_matches",
        type=int,
        default=REALTIME_MAX_MATCH_DRAW,
        help=f"实时左图最多绘制多少条匹配，默认 {REALTIME_MAX_MATCH_DRAW}",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    image_dir = os.path.abspath(args.image_dir) if args.image_dir else os.path.join(data_dir, IMAGE_DIR_NAME)
    out_path = args.out or os.path.join(data_dir, DEFAULT_OUT_NAME)

    frames = list_image_frames(
        image_dir,
        max_frames=args.max_frames,
        step=max(1, args.step),
        start_ratio=args.start_ratio,
        end_ratio=args.end_ratio,
    )
    odom_path = os.path.join(data_dir, ODOM_FILE_NAME)
    odom_velocity = load_odom_velocity(odom_path) if args.scale_mode == "wheel" else None
    if args.scale_mode == "wheel" and odom_velocity is None:
        print(f"警告: 未找到 {odom_path}，已退化为 unit scale", file=sys.stderr)

    print(
        f"images={len(frames)}, image_dir={image_dir}, "
        f"range={args.start_ratio:.2f}%~{args.end_ratio:.2f}%, step={max(1, args.step)}"
    )
    print(f"scale_mode={args.scale_mode}, wheel_scale_available={odom_velocity is not None}")
    t, pos, quat, stats = run_visual_odometry_realtime(
        frames,
        odom_velocity,
        args.scale_mode,
        pause_sec=args.realtime_pause,
        max_match_draw=args.realtime_matches,
    )
    save_pose_csv(out_path, t, pos, quat)

    print(f"已写入 {len(t)} 帧 -> {out_path}")
    print(f"末位置 [x,y,z]: {pos[-1]}")
    print(
        "VO stats: "
        f"pairs={stats.total_pairs}, used={stats.used_pairs}, "
        f"skip_matches={stats.skipped_matches}, skip_essential={stats.skipped_essential}, "
        f"skip_inliers={stats.skipped_inliers}"
    )

if __name__ == "__main__":
    main()
