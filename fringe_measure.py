#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从实验数据或图片估算衍射条纹间距。

作者：韩东杰
学号：3260103072

支持三种输入：
    1. --csv：一维强度数据，列为 position_mm,intensity 或 pixel,intensity
    2. --image：实拍灰度或彩色图片，取中央横带的平均剖面
    3. 不带输入：使用内置合成剖面，仅用于调试

用法：
    python fringe_measure.py
    python fringe_measure.py --csv data/fringe_profile.csv --expected-mm 1.30
    python fringe_measure.py --image 衍射照片.png --pixel-mm 0.042 --expected-mm 1.30
    python fringe_measure.py --csv data/fringe_profile.csv --output output/charts/my_profile.png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "charts")


def configure_stream(stream) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass


def synthetic_profile(x_mm: np.ndarray, spacing_mm: float) -> np.ndarray:
    """生成带噪声的双缝剖面，仅用于测试找峰逻辑。"""
    wavelength_m = 650e-9
    slit_width_m = 100e-6
    distance_m = 1.0
    x_m = x_mm * 1e-3
    beta = np.pi * slit_width_m * x_m / (wavelength_m * distance_m)
    alpha = np.pi * x_m / (spacing_mm * 1e-3)
    with np.errstate(divide="ignore", invalid="ignore"):
        envelope = (np.sin(beta) / beta) ** 2
    envelope = np.where(np.isclose(beta, 0.0, atol=1e-12), 1.0, envelope)
    signal = (np.cos(alpha) ** 2) * envelope
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 0.018, signal.shape)
    background = 0.015 + 0.004 * x_mm / (np.max(np.abs(x_mm)) or 1.0)
    return signal + background + noise


def load_image_profile(image_path: str, band_fraction: float = 0.30) -> np.ndarray:
    """读取灰度图，扣背景，取中央横带的平均剖面。"""
    image = plt.imread(image_path).astype(float)
    if image.ndim == 3:
        image = image.mean(axis=2)
    height, width = image.shape
    edge = min(10, width // 10)
    background = float(np.median(np.concatenate([image[:, :edge], image[:, -edge:]], axis=1)))
    image = np.clip(image - background, 0.0, None)
    band = max(1, int(height * band_fraction))
    start = (height - band) // 2
    profile = image[start:start + band, :].mean(axis=0)
    return (profile - profile.min()) / (profile.max() - profile.min() + 1e-12)


def load_csv_profile(path: str) -> tuple[np.ndarray, np.ndarray, str]:
    """读取 position_mm/intensity 或 pixel/intensity 两列数据。"""
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = {name.strip().lower(): name for name in (reader.fieldnames or [])}
        position_key = next(
            (key for key in ("position_mm", "x_mm", "position", "x", "pixel") if key in fields),
            None,
        )
        intensity_key = next(
            (key for key in ("intensity", "power", "signal") if key in fields),
            None,
        )
        if position_key is None or intensity_key is None:
            raise ValueError(
                f"{path} 需要 position_mm/intensity 或 pixel/intensity 两列，"
                f"当前列为 {list(fields)}"
            )
        position_field = fields[position_key]
        intensity_field = fields[intensity_key]
        x_values: list[float] = []
        y_values: list[float] = []
        for row_number, row in enumerate(reader, start=2):
            if not row.get(position_field) or not row.get(intensity_field):
                continue
            try:
                x_values.append(float(row[position_field]))
                y_values.append(float(row[intensity_field]))
            except ValueError as exc:
                raise ValueError(f"{path} 第 {row_number} 行不是有效数字") from exc

    if len(x_values) < 5:
        raise ValueError(f"{path} 至少需要 5 行强度数据")
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    y = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-12)
    unit = "pixel" if position_key == "pixel" else "mm"
    return x, y, unit


def smooth(values: np.ndarray, width: int) -> np.ndarray:
    kernel = np.ones(width, dtype=float) / width
    return np.convolve(values, kernel, mode="same")


def find_peaks(values: np.ndarray, min_height: float) -> np.ndarray:
    sign_change = np.diff(np.sign(np.diff(values)))
    peaks = np.where(sign_change < 0)[0] + 1
    return peaks[(values[peaks] >= min_height) & (peaks > 0) & (peaks < values.size - 1)]


def filter_close_peaks(values: np.ndarray, indices: np.ndarray, min_distance: int) -> np.ndarray:
    """按高度从高到低保留彼此距离不小于 min_distance 的峰。"""
    order = np.argsort(values[indices])[::-1]
    selected: list[int] = []
    for position in order:
        candidate = int(indices[position])
        if all(abs(candidate - kept) >= min_distance for kept in selected):
            selected.append(candidate)
    return np.sort(np.asarray(selected, dtype=int))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="估算衍射条纹间距")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", type=str, default=None, help="一维强度 CSV 文件")
    source.add_argument("--image", type=str, default=None, help="实拍灰度或彩色图片路径")
    parser.add_argument("--pixel-mm", type=float, default=None, help="每像素对应的毫米数")
    parser.add_argument("--expected-mm", type=float, default=1.30, help="理论条纹间距，单位 mm")
    parser.add_argument("--smooth-width", type=int, default=15, help="滑动平均窗口宽度")
    parser.add_argument("--output", type=str, default=None, help="输出 PNG 路径")
    return parser.parse_args()


def main() -> int:
    configure_stream(sys.stdout)
    args = parse_args()

    if args.csv:
        x_values, profile, unit = load_csv_profile(args.csv)
        if unit == "pixel" and args.pixel_mm:
            x_values = x_values * args.pixel_mm
            unit = "mm"
        source_label = os.path.basename(args.csv)
        scale_note = (
            f"标定：{args.pixel_mm} mm/pixel"
            if args.pixel_mm
            else "CSV 坐标单位：" + unit
        )
    elif args.image:
        profile = load_image_profile(args.image)
        if args.pixel_mm:
            x_values = np.arange(profile.size) * args.pixel_mm
            unit = "mm"
            scale_note = f"标定：{args.pixel_mm} mm/pixel"
        else:
            x_values = np.arange(profile.size, dtype=float)
            unit = "pixel"
            scale_note = "未标定，结果单位为像素"
        source_label = os.path.basename(args.image)
    else:
        x_values = np.linspace(-15.0, 15.0, 3001)
        profile = synthetic_profile(x_values, args.expected_mm)
        unit = "mm"
        source_label = "内置合成剖面（仅用于演示）"
        scale_note = "合成数据，坐标为 mm"

    smoothed = smooth(profile, max(3, args.smooth_width))
    peak_indices = find_peaks(smoothed, float(np.max(smoothed) * 0.30))

    if unit == "mm":
        median_step = float(np.median(np.diff(x_values)))
        min_separation = max(3, int(round(args.expected_mm / median_step * 0.35)))
    elif args.pixel_mm:
        min_separation = max(3, int(round(args.expected_mm / args.pixel_mm * 0.35)))
    else:
        min_separation = max(3, smoothed.size // 40)

    peak_indices = filter_close_peaks(smoothed, peak_indices, min_separation)
    peak_x = x_values[peak_indices]
    peak_y = smoothed[peak_indices]

    if peak_x.size < 2:
        print("没有找到足够多的条纹峰，先检查过曝、背景和图片方向。")
        return 1

    gaps = np.diff(peak_x)
    median_gap = float(np.median(gaps))
    kept = gaps[np.abs(gaps - median_gap) <= 0.25 * median_gap] if median_gap else gaps
    if kept.size < 2:
        kept = gaps
    mean_gap = float(np.mean(kept))
    if kept.size >= 2:
        std_gap = float(np.std(kept, ddof=1))
        std_error = std_gap / np.sqrt(kept.size)
    else:
        std_error = 0.0

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_values, smoothed, color="#1D4ED8", linewidth=1.2, label="平滑后的剖面")
    ax.scatter(peak_x, peak_y, color="#B45309", s=28, zorder=3, label="识别到的峰")
    ax.set_xlabel(f"屏幕位置（{unit}）")
    ax.set_ylabel("归一化强度")
    ax.set_title(f"衍射条纹峰识别：{source_label}")
    ax.grid(color="#E3E9EE", linewidth=0.8)
    ax.legend()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    output_path = args.output or os.path.join(OUTPUT_DIR, "fringe_measurement.png")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"数据来源：{source_label}")
    print(scale_note)
    print(f"找到峰：{peak_x.size} 个")
    print(f"使用间距：{kept.size} 个")
    print(f"条纹间距 = {mean_gap:.4f} ± {std_error:.4f} {unit}")
    print(f"理论间距 = {args.expected_mm:.4f} mm")
    if unit == "mm":
        relative = (mean_gap - args.expected_mm) / args.expected_mm * 100.0
        print(f"相对偏差 = {relative:+.2f}%")
    else:
        print("提示：没有给 --pixel-mm，当前结果是像素间距。")
    print(f"图已保存：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
