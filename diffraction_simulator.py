#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""夫琅禾费衍射数值模拟。

计算单缝、双缝、多缝在远处屏幕上的强度分布，使用小角度近似
sin(theta) ≈ x / L。只依赖 numpy 和 matplotlib。

用法：
    python diffraction_simulator.py
    python diffraction_simulator.py --wavelength-nm 650 --slit-um 100 --separation-um 500
"""

from __future__ import annotations

import argparse
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
    """设置控制台编码，旧接口缺失时静默跳过。"""
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass


def sinc(beta: np.ndarray) -> np.ndarray:
    """返回 sin(beta) / beta，beta 为 0 时返回 1。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.sin(beta) / beta
    return np.where(np.isclose(beta, 0.0, atol=1e-12), 1.0, value)


def single_slit_intensity(
    x_m: np.ndarray,
    wavelength_m: float,
    slit_width_m: float,
    distance_m: float,
) -> np.ndarray:
    beta = np.pi * slit_width_m * x_m / (wavelength_m * distance_m)
    return sinc(beta) ** 2


def double_slit_intensity(
    x_m: np.ndarray,
    wavelength_m: float,
    slit_width_m: float,
    separation_m: float,
    distance_m: float,
) -> np.ndarray:
    beta = np.pi * slit_width_m * x_m / (wavelength_m * distance_m)
    alpha = np.pi * separation_m * x_m / (wavelength_m * distance_m)
    return (np.cos(alpha) ** 2) * (sinc(beta) ** 2)


def n_slit_intensity(
    x_m: np.ndarray,
    wavelength_m: float,
    slit_width_m: float,
    separation_m: float,
    count: int,
    distance_m: float,
) -> np.ndarray:
    beta = np.pi * slit_width_m * x_m / (wavelength_m * distance_m)
    phase = 2.0 * np.pi * separation_m * x_m / (wavelength_m * distance_m)
    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = np.sin(count * phase / 2.0)
        denominator = np.sin(phase / 2.0)
        grating = (numerator / denominator) ** 2
    grating = np.where(np.isclose(denominator, 0.0, atol=1e-12), float(count**2), grating)
    return grating * (sinc(beta) ** 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="夫琅禾费衍射模拟")
    parser.add_argument("--wavelength-nm", type=float, default=650.0, help="激光波长，单位 nm")
    parser.add_argument("--slit-um", type=float, default=100.0, help="单缝宽度，单位 μm")
    parser.add_argument("--separation-um", type=float, default=500.0, help="缝间距，单位 μm")
    parser.add_argument("--count", type=int, default=5, help="多缝数量")
    parser.add_argument("--distance-m", type=float, default=1.0, help="缝到屏幕距离，单位 m")
    parser.add_argument("--screen-mm", type=float, default=25.0, help="屏幕半宽，单位 mm")
    return parser.parse_args()


def main() -> int:
    configure_stream(sys.stdout)
    args = parse_args()

    wavelength_m = args.wavelength_nm * 1e-9
    slit_width_m = args.slit_um * 1e-6
    separation_m = args.separation_um * 1e-6

    x_mm = np.linspace(-args.screen_mm, args.screen_mm, 4001)
    x_m = x_mm * 1e-3

    single = single_slit_intensity(x_m, wavelength_m, slit_width_m, args.distance_m)
    double = double_slit_intensity(x_m, wavelength_m, slit_width_m, separation_m, args.distance_m)
    multi = n_slit_intensity(x_m, wavelength_m, slit_width_m, separation_m, args.count, args.distance_m)
    single /= np.max(single)
    double /= np.max(double)
    multi /= np.max(multi)

    fringe_spacing_mm = wavelength_m * args.distance_m / separation_m * 1e3
    first_minimum_mm = wavelength_m * args.distance_m / slit_width_m * 1e3

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    colors = ["#0F766E", "#1D4ED8", "#B45309"]
    data = [
        (single, f"单缝，a={args.slit_um:g} μm"),
        (double, f"双缝，a={args.slit_um:g} μm，d={args.separation_um:g} μm"),
        (multi, f"{args.count} 缝，a={args.slit_um:g} μm，d={args.separation_um:g} μm"),
    ]
    for ax, (values, label), color in zip(axes, data, colors):
        ax.fill_between(x_mm, values, color=color, alpha=0.55, linewidth=0)
        ax.set_ylabel("归一化强度")
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.text(0.02, 0.86, label, transform=ax.transAxes, fontsize=10, color="#10263B")
        ax.grid(axis="x", color="#E3E9EE", linewidth=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[-1].set_xlabel("屏幕位置 x (mm)")
    axes[0].set_title(
        f"夫琅禾费衍射强度分布，λ={args.wavelength_nm:g} nm，L={args.distance_m:g} m",
        fontsize=12,
        color="#10263B",
    )
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "diffraction_patterns.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"波长 λ = {args.wavelength_nm:g} nm")
    print(f"缝宽 a = {args.slit_um:g} μm")
    print(f"缝间距 d = {args.separation_um:g} μm")
    print(f"屏距 L = {args.distance_m:g} m")
    print(f"双缝条纹间距 Δx = {fringe_spacing_mm:.3f} mm")
    print(f"单缝第一暗纹位置 x1 = ±{first_minimum_mm:.3f} mm")
    print(f"图已保存：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
