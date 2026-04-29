#!/usr/bin/env python3
"""
光流米制尺度标定。

输入每组实验的：
- cum_x / cum_y: 一段运动结束后的光流累积量（flow 原始单位）
- height_m: 光流传感器到地面的高度（m）
- distance_m: 真实测量运动距离（m）

输出：
- 每组实验的米制尺度：meters_per_unit = distance / hypot(cum_x, cum_y)
- 高度归一化尺度：meters_per_unit_per_meter_height = meters_per_unit / height
- 全局常数高度模型：meters_per_unit ≈ k * height
- 由高度模型预测的旋转尺度：flow_units_per_rad ≈ height / meters_per_unit = 1 / k

如果你有带方向的真实位移，也可以在 CSV 中额外提供 true_x_m / true_y_m。
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CalibrationSample:
    name: str
    cum_x: float
    cum_y: float
    height_m: float
    distance_m: float
    true_x_m: float | None = None
    true_y_m: float | None = None


# 前向运动测量数据
CALIBRATION_SAMPLES = [
  CalibrationSample("forward_1", cum_x=-1050.0, cum_y=6881.0, height_m=0.06, distance_m=0.685),
  CalibrationSample("forward_2", cum_x=-564.0, cum_y=3516.0, height_m=0.061, distance_m=0.355),
  CalibrationSample("forward_3", cum_x=-150.0, cum_y=1073.0, height_m=0.062, distance_m=0.111),
  CalibrationSample("forward_4", cum_x=-799.0, cum_y=5055.0, height_m=0.0597, distance_m=0.507),
  CalibrationSample("forward_5", cum_x=-201.0, cum_y=1619.0, height_m=0.0611, distance_m=0.171),
  CalibrationSample("forward_6", cum_x=-441.0, cum_y=2699.0, height_m=0.0614, distance_m=0.266),
  CalibrationSample("forward_7", cum_x=-512.0, cum_y=3065.0, height_m=0.0609, distance_m=0.305),
]


def _finite_positive(value: float, name: str, sample_name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{sample_name}: {name} 必须为正有限数，当前 {value!r}")


def validate_samples(samples: Iterable[CalibrationSample]) -> list[CalibrationSample]:
    out = list(samples)
    if not out:
        raise ValueError("没有标定数据：请填 CALIBRATION_SAMPLES 或使用 --csv")
    for s in out:
        flow_norm = float(np.hypot(s.cum_x, s.cum_y))
        _finite_positive(flow_norm, "sqrt(cum_x^2 + cum_y^2)", s.name)
        _finite_positive(s.height_m, "height_m", s.name)
        _finite_positive(s.distance_m, "distance_m", s.name)
    return out


def load_samples_csv(path: str) -> list[CalibrationSample]:
    """
    CSV 至少包含列：cum_x,cum_y,height_m,distance_m。
    可选列：name,true_x_m,true_y_m。
    """
    samples: list[CalibrationSample] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 没有表头: {path}")
        required = {"cum_x", "cum_y", "height_m", "distance_m"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV 缺少列 {sorted(missing)}；需要 {sorted(required)}")
        for idx, row in enumerate(reader, start=1):
            name = (row.get("name") or f"sample_{idx}").strip()
            true_x = row.get("true_x_m")
            true_y = row.get("true_y_m")
            samples.append(
                CalibrationSample(
                    name=name,
                    cum_x=float(row["cum_x"]),
                    cum_y=float(row["cum_y"]),
                    height_m=float(row["height_m"]),
                    distance_m=float(row["distance_m"]),
                    true_x_m=float(true_x) if true_x not in (None, "") else None,
                    true_y_m=float(true_y) if true_y not in (None, "") else None,
                )
            )
    return samples


def fit_constant_scale(samples: list[CalibrationSample]) -> float:
    """拟合 distance ≈ scale * hypot(cum_x,cum_y)。"""
    flow_norm = np.array([np.hypot(s.cum_x, s.cum_y) for s in samples], dtype=np.float64)
    distance = np.array([s.distance_m for s in samples], dtype=np.float64)
    return float(np.dot(flow_norm, distance) / np.dot(flow_norm, flow_norm))


def fit_height_scale_k(samples: list[CalibrationSample]) -> float:
    """拟合 distance ≈ k * height * hypot(cum_x,cum_y)，返回 k = meters_per_unit_per_meter_height。"""
    x = np.array([s.height_m * np.hypot(s.cum_x, s.cum_y) for s in samples], dtype=np.float64)
    y = np.array([s.distance_m for s in samples], dtype=np.float64)
    return float(np.dot(x, y) / np.dot(x, x))


def fit_axis_scales_if_available(samples: list[CalibrationSample]) -> tuple[float | None, float | None]:
    """
    如果提供 true_x_m/true_y_m，则拟合 true_x ≈ sx*cum_x, true_y ≈ sy*cum_y。
    注意这里使用有符号位移，坐标正负需与 cum_x/cum_y 一致。
    """
    xs = [s for s in samples if s.true_x_m is not None and abs(s.cum_x) > 1e-12]
    ys = [s for s in samples if s.true_y_m is not None and abs(s.cum_y) > 1e-12]
    sx = None
    sy = None
    if xs:
        a = np.array([s.cum_x for s in xs], dtype=np.float64)
        b = np.array([s.true_x_m for s in xs], dtype=np.float64)
        sx = float(np.dot(a, b) / np.dot(a, a))
    if ys:
        a = np.array([s.cum_y for s in ys], dtype=np.float64)
        b = np.array([s.true_y_m for s in ys], dtype=np.float64)
        sy = float(np.dot(a, b) / np.dot(a, a))
    return sx, sy


def print_report(samples: list[CalibrationSample]) -> None:
    samples = validate_samples(samples)
    print("Per-sample scale:")
    print(
        "name,cum_x,cum_y,flow_norm,height_m,distance_m,meters_per_unit,"
        "meters_per_unit_per_meter_height,pred_rot_gain_flow_units_per_rad"
    )
    per_scales = []
    per_k = []
    for s in samples:
        flow_norm = float(np.hypot(s.cum_x, s.cum_y))
        scale = s.distance_m / flow_norm
        k_i = scale / s.height_m
        rot_gain_i = s.height_m / scale
        per_scales.append(scale)
        per_k.append(k_i)
        print(
            f"{s.name},{s.cum_x:.9g},{s.cum_y:.9g},{flow_norm:.9g},"
            f"{s.height_m:.9g},{s.distance_m:.9g},{scale:.9g},{k_i:.9g},{rot_gain_i:.9g}"
        )

    per_scales_arr = np.array(per_scales, dtype=np.float64)
    per_k_arr = np.array(per_k, dtype=np.float64)
    scale_const = fit_constant_scale(samples)
    k_height = fit_height_scale_k(samples)
    rot_gain_from_height = 1.0 / k_height
    sx, sy = fit_axis_scales_if_available(samples)

    print("\nSummary:")
    print(f"constant meters_per_unit (least squares): {scale_const:.9g}")
    print(f"per-sample meters_per_unit mean/std: {np.mean(per_scales_arr):.9g} / {np.std(per_scales_arr):.9g}")
    print(f"height model k = meters_per_unit / height: {k_height:.9g}")
    print(f"per-sample k mean/std: {np.mean(per_k_arr):.9g} / {np.std(per_k_arr):.9g}")
    print(f"predicted rotation gain from height model (flow_units/rad): {rot_gain_from_height:.9g}")
    print("\nSuggested constants:")
    print(f"FLOW_METERS_PER_UNIT = {scale_const:.12g}  # 若高度变化很小，用这个")
    print(f"FLOW_METERS_PER_UNIT_PER_METER_HEIGHT = {k_height:.12g}  # meters/unit = k * height_m")
    print(f"FLOW_ROT_GAIN_X = {rot_gain_from_height:.12g}")
    print(f"FLOW_ROT_GAIN_Y = {rot_gain_from_height:.12g}")
    if sx is not None or sy is not None:
        print("\nAxis scales from signed true_x_m/true_y_m:")
        if sx is not None:
            print(f"FLOW_METERS_PER_UNIT_X = {sx:.12g}")
        if sy is not None:
            print(f"FLOW_METERS_PER_UNIT_Y = {sy:.12g}")


def main() -> None:
    parser = argparse.ArgumentParser(description="根据多组 flow 累积量与真实距离标定米制尺度")
    parser.add_argument(
        "--csv",
        default="",
        help="标定 CSV：列 cum_x,cum_y,height_m,distance_m；可选 name,true_x_m,true_y_m",
    )
    args = parser.parse_args()

    samples = load_samples_csv(args.csv) if args.csv else CALIBRATION_SAMPLES
    try:
        print_report(samples)
    except ValueError as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
