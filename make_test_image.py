#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成合成衍射条纹照片和一维强度 CSV。

作者：韩东杰
学号：3260103072

这些数据只用于在没有实验器材时验证图片读取和找峰流程，
不能当作真实实验数据。

用法：
    python make_test_image.py
    python make_test_image.py --spacing-mm 1.30 --pixel-mm 0.02
    python make_test_image.py --profile-csv data/fringe_profile.csv
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def configure_stream(stream) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass


def double_slit_image(
    width_px: int,
    height_px: int,
    pixel_mm: float,
    spacing_mm: float,
    noise_level: float,
) -> np.ndarray:
    x_mm = (np.arange(width_px) - width_px / 2.0) * pixel_mm
    x_m = x_mm * 1e-3
    wavelength_m = 650e-9
    slit_width_m = 100e-6
    distance_m = 1.0

    beta = np.pi * slit_width_m * x_m / (wavelength_m * distance_m)
    alpha = np.pi * x_m / (spacing_mm * 1e-3)
    with np.errstate(divide="ignore", invalid="ignore"):
        envelope = (np.sin(beta) / beta) ** 2
    envelope = np.where(np.isclose(beta, 0.0, atol=1e-12), 1.0, envelope)
    signal = (np.cos(alpha) ** 2) * envelope

    rng = np.random.default_rng(11)
    image = np.tile(signal, (height_px, 1))
    noise = rng.normal(0.0, noise_level, image.shape)
    return np.clip(image + 0.03 + noise, 0.0, 1.0)


def write_profile_csv(image: np.ndarray, pixel_mm: float, path: str) -> None:
    row = image[image.shape[0] // 2, :]
    x_mm = (np.arange(row.size) - row.size / 2.0) * pixel_mm
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["position_mm", "intensity"])
        for x_value, y_value in zip(x_mm, row):
            writer.writerow([f"{x_value:.6f}", f"{y_value:.6f}"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成合成衍射测试数据")
    parser.add_argument("--width", type=int, default=1200, help="图像宽度（像素）")
    parser.add_argument("--height", type=int, default=800, help="图像高度（像素）")
    parser.add_argument("--pixel-mm", type=float, default=0.02, help="每像素毫米数")
    parser.add_argument("--spacing-mm", type=float, default=1.30, help="理论条纹间距（mm）")
    parser.add_argument("--noise", type=float, default=0.025, help="高斯噪声标准差")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(DATA_DIR, "synthetic_diffraction.png"),
        help="输出图片路径",
    )
    parser.add_argument("--profile-csv", type=str, default=None, help="可选：同时输出一维强度 CSV")
    return parser.parse_args()


def main() -> int:
    configure_stream(sys.stdout)
    args = parse_args()
    image = double_slit_image(
        args.width,
        args.height,
        args.pixel_mm,
        args.spacing_mm,
        args.noise,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.imsave(args.output, image, cmap="gray", vmin=0.0, vmax=1.0)
    print(f"合成照片已保存：{args.output}")
    print(f"像素标定：{args.pixel_mm} mm/pixel")
    print(f"理论条纹间距：{args.spacing_mm} mm")
    print("该图片是合成数据，不能写入实验报告。")

    if args.profile_csv:
        write_profile_csv(image, args.pixel_mm, args.profile_csv)
        print(f"一维强度 CSV 已保存：{args.profile_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
