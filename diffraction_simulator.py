#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""夫琅禾费衍射数值模拟。

作者：韩东杰
学号：3260103072

计算单缝、双缝、多缝在远处屏幕上的强度分布，使用小角度近似
sin(theta) ≈ x / L。只依赖 numpy 和 matplotlib。

用法：
    python diffraction_simulator.py
    python diffraction_simulator.py --wavelength-nm 650 --slit-um 100 --separation-um 500
    python diffraction_simulator.py --input-csv data/diffraction_cases.csv
    python diffraction_simulator.py --input-csv data/diffraction_cases.csv --output output/charts/my_cases.png

CSV 必需列：
    wavelength_nm, slit_um, separation_um

CSV 可选列：
    label, count, distance_m, screen_mm
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "charts")


@dataclass(frozen=True)
class DiffractionCase:
    label: str
    wavelength_nm: float
    slit_um: float
    separation_um: float
    count: int = 2
    distance_m: float = 1.0
    screen_mm: float = 25.0


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


def read_cases(path: str) -> list[DiffractionCase]:
    """从 CSV 读取实验参数，每一行生成一张子图。"""
    required = {"wavelength_nm", "slit_um", "separation_um"}
    cases: list[DiffractionCase] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = {name.strip() for name in (reader.fieldnames or [])}
        missing = required - fields
        if missing:
            raise ValueError(f"CSV 缺少列：{', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            try:
                case = DiffractionCase(
                    label=(row.get("label") or f"第 {row_number - 1} 组").strip(),
                    wavelength_nm=float(row["wavelength_nm"]),
                    slit_um=float(row["slit_um"]),
                    separation_um=float(row["separation_um"]),
                    count=int(float(row.get("count") or 2)),
                    distance_m=float(row.get("distance_m") or 1.0),
                    screen_mm=float(row.get("screen_mm") or 25.0),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path} 第 {row_number} 行格式错误：{exc}") from exc
            if case.wavelength_nm <= 0 or case.slit_um <= 0 or case.separation_um <= 0:
                raise ValueError(f"{path} 第 {row_number} 行存在非正数参数")
            if case.count < 1 or case.distance_m <= 0 or case.screen_mm <= 0:
                raise ValueError(f"{path} 第 {row_number} 行存在无效的 count、distance_m 或 screen_mm")
            cases.append(case)
    if not cases:
        raise ValueError(f"{path} 中没有数据行")
    return cases


def compute_case(case: DiffractionCase) -> tuple[np.ndarray, np.ndarray, float, float]:
    wavelength_m = case.wavelength_nm * 1e-9
    slit_width_m = case.slit_um * 1e-6
    separation_m = case.separation_um * 1e-6
    x_mm = np.linspace(-case.screen_mm, case.screen_mm, 4001)
    x_m = x_mm * 1e-3

    if case.count == 1:
        intensity = single_slit_intensity(x_m, wavelength_m, slit_width_m, case.distance_m)
    elif case.count == 2:
        intensity = double_slit_intensity(
            x_m, wavelength_m, slit_width_m, separation_m, case.distance_m
        )
    else:
        intensity = n_slit_intensity(
            x_m, wavelength_m, slit_width_m, separation_m, case.count, case.distance_m
        )

    peak = float(np.max(intensity))
    if peak <= 0:
        raise ValueError(f"{case.label} 的强度全为 0，检查参数")
    intensity = intensity / peak
    fringe_spacing_mm = wavelength_m * case.distance_m / separation_m * 1e3
    first_minimum_mm = wavelength_m * case.distance_m / slit_width_m * 1e3
    return x_mm, intensity, fringe_spacing_mm, first_minimum_mm


def plot_cases(cases: list[DiffractionCase], output_path: str) -> None:
    colors = ["#0F766E", "#1D4ED8", "#B45309", "#7C3AED", "#BE123C"]
    fig, axes = plt.subplots(
        len(cases),
        1,
        figsize=(9, max(3.4, 2.35 * len(cases))),
        squeeze=False,
    )
    for index, case in enumerate(cases):
        x_mm, intensity, _, _ = compute_case(case)
        ax = axes[index, 0]
        color = colors[index % len(colors)]
        ax.fill_between(x_mm, intensity, color=color, alpha=0.55, linewidth=0)
        ax.set_ylabel("归一化强度")
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.text(
            0.02,
            0.86,
            f"{case.label}：λ={case.wavelength_nm:g} nm，a={case.slit_um:g} μm，"
            f"d={case.separation_um:g} μm，N={case.count}",
            transform=ax.transAxes,
            fontsize=9.5,
            color="#10263B",
        )
        ax.grid(axis="x", color="#E3E9EE", linewidth=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[-1, 0].set_xlabel("屏幕位置 x (mm)")
    fig.suptitle(f"夫琅禾费衍射：{len(cases)} 组实验参数", fontsize=13, color="#10263B")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="夫琅禾费衍射模拟")
    parser.add_argument("--input-csv", type=str, default=None, help="实验参数 CSV 文件")
    parser.add_argument("--output", type=str, default=None, help="输出 PNG 路径")
    parser.add_argument("--wavelength-nm", type=float, default=650.0, help="激光波长，单位 nm")
    parser.add_argument("--slit-um", type=float, default=100.0, help="单缝宽度，单位 μm")
    parser.add_argument("--separation-um", type=float, default=500.0, help="缝间距，单位 μm")
    parser.add_argument("--count", type=int, default=5, help="多缝数量")
    parser.add_argument("--distance-m", type=float, default=1.0, help="缝到屏幕距离，单位 m")
    parser.add_argument("--screen-mm", type=float, default=25.0, help="屏幕半宽，单位 mm")
    return parser.parse_args()


def default_cases(args: argparse.Namespace) -> list[DiffractionCase]:
    base = dict(
        wavelength_nm=args.wavelength_nm,
        slit_um=args.slit_um,
        separation_um=args.separation_um,
        distance_m=args.distance_m,
        screen_mm=args.screen_mm,
    )
    return [
        DiffractionCase(label="单缝", count=1, **base),
        DiffractionCase(label="双缝", count=2, **base),
        DiffractionCase(label=f"{args.count} 缝", count=args.count, **base),
    ]


def main() -> int:
    configure_stream(sys.stdout)
    args = parse_args()

    if args.input_csv:
        cases = read_cases(args.input_csv)
        output_path = args.output or os.path.join(OUTPUT_DIR, "diffraction_cases.png")
    else:
        cases = default_cases(args)
        output_path = args.output or os.path.join(OUTPUT_DIR, "diffraction_patterns.png")

    plot_cases(cases, output_path)
    print(f"共 {len(cases)} 组参数")
    for case in cases:
        _, _, fringe_spacing_mm, first_minimum_mm = compute_case(case)
        print(
            f"[{case.label}] λ={case.wavelength_nm:g} nm，a={case.slit_um:g} μm，"
            f"d={case.separation_um:g} μm，N={case.count}，L={case.distance_m:g} m"
        )
        print(f"  单缝第一暗纹：±{first_minimum_mm:.3f} mm")
        if case.count >= 2:
            print(f"  双缝/多缝条纹间距：{fringe_spacing_mm:.3f} mm")
    print(f"图已保存：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
