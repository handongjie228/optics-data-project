#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成合成衍射条纹照片。

这张图只用于在没有实验器材时验证 fringe_measure.py 的图片读取和找峰
流程，不能当作真实实验数据。

生成后可用：
    python fringe_measure.py --image data/synthetic_diffraction.png --pixel-mm 0.02 --expected-mm 1.30
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data")


def configure_stream(stream) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass


def double_slit_image(width_px: int, height_px: int, pixel_mm: float, spacing_mm: float) -> np.ndarray:
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
    noise = rng.normal(0.0, 0.025, image.shape)
    return np.clip(image + 0.03 + noise, 0.0, 1.0)


def main() -> int:
    configure_stream(sys.stdout)
    width, height, pixel_mm, spacing_mm = 1200, 800, 0.02, 1.30
    image = double_slit_image(width, height, pixel_mm, spacing_mm)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "synthetic_diffraction.png")
    plt.imsave(output_path, image, cmap="gray", vmin=0.0, vmax=1.0)

    print(f"合成照片已保存：{output_path}")
    print(f"像素标定：{pixel_mm} mm/pixel")
    print(f"理论条纹间距：{spacing_mm} mm")
    print("该图片是合成数据，不能写入实验报告。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
