#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最小参数反演示例：从一维衍射强度曲线预测条纹间距。

作者：韩东杰
学号：3260103072

这是一个教学示例，不是研究成果：
1. 用双缝公式生成不同条纹间距的合成曲线。
2. 用 k 近邻（k-NN）学习“曲线 -> 条纹间距”的映射。
3. 在未参与训练的合成曲线上评估预测误差。

只依赖 numpy 和 matplotlib。

用法：
    python ml_inversion_demo.py
    python ml_inversion_demo.py --samples 400 --neighbors 7
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
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass


def sinc(beta: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.sin(beta) / beta
    return np.where(np.isclose(beta, 0.0, atol=1e-12), 1.0, value)


def make_profile(spacing_mm: float, x_mm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """生成一条带噪声的双缝衍射强度曲线。"""
    wavelength_m = 650e-9
    slit_width_m = 100e-6
    distance_m = 1.0
    x_m = x_mm * 1e-3
    beta = np.pi * slit_width_m * x_m / (wavelength_m * distance_m)
    alpha = np.pi * x_m / (spacing_mm * 1e-3)
    envelope = sinc(beta) ** 2
    signal = (np.cos(alpha) ** 2) * envelope
    amplitude = rng.uniform(0.85, 1.15)
    background = rng.uniform(0.01, 0.04)
    noise = rng.normal(0.0, 0.018, signal.shape)
    return np.clip(amplitude * signal + background + noise, 0.0, None)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    minimum = matrix.min(axis=1, keepdims=True)
    maximum = matrix.max(axis=1, keepdims=True)
    return (matrix - minimum) / (maximum - minimum + 1e-12)


def knn_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    predictions = []
    for sample in x_test:
        distances = np.linalg.norm(x_train - sample, axis=1)
        nearest = np.argpartition(distances, neighbors - 1)[:neighbors]
        weights = 1.0 / (distances[nearest] + 1e-6)
        predictions.append(float(np.average(y_train[nearest], weights=weights)))
    return np.asarray(predictions)


def make_dataset(samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_mm = np.linspace(-12.0, 12.0, 1201)
    spacings = rng.uniform(0.8, 2.0, samples)
    profiles = np.stack([make_profile(spacing, x_mm, rng) for spacing in spacings])
    return profiles[:, ::4], spacings, x_mm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="衍射条纹间距的机器学习反演示例")
    parser.add_argument("--samples", type=int, default=320, help="合成样本数量")
    parser.add_argument("--neighbors", type=int, default=5, help="k-NN 的邻居数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main() -> int:
    configure_stream(sys.stdout)
    args = parse_args()
    features, targets, _ = make_dataset(max(80, args.samples), args.seed)
    features = normalize_rows(features)

    rng = np.random.default_rng(args.seed + 1)
    indices = rng.permutation(len(features))
    split = int(len(features) * 0.75)
    train_idx = indices[:split]
    test_idx = indices[split:]
    x_train, y_train = features[train_idx], targets[train_idx]
    x_test, y_test = features[test_idx], targets[test_idx]

    predictions = knn_predict(x_train, y_train, x_test, max(1, args.neighbors))
    errors = predictions - y_test
    mae = float(np.mean(np.abs(errors)))
    relative = float(np.mean(np.abs(errors) / y_test) * 100.0)

    sizes = [size for size in (40, 80, 120, 160, 200, len(x_train)) if size <= len(x_train)]
    sizes = sorted(set(sizes))
    curve_errors = []
    for size in sizes:
        subset = rng.choice(len(x_train), size=size, replace=False)
        subset_predictions = knn_predict(
            x_train[subset],
            y_train[subset],
            x_test,
            max(1, args.neighbors),
        )
        curve_errors.append(float(np.mean(np.abs(subset_predictions - y_test))))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].scatter(y_test, predictions, color="#0F766E", alpha=0.75, s=22)
    low = min(float(y_test.min()), float(predictions.min()))
    high = max(float(y_test.max()), float(predictions.max()))
    axes[0].plot([low, high], [low, high], color="#B45309", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("真实条纹间距（mm）")
    axes[0].set_ylabel("模型预测（mm）")
    axes[0].set_title(f"测试集：MAE={mae:.4f} mm，平均相对误差={relative:.2f}%")
    axes[0].grid(color="#E3E9EE", linewidth=0.8)

    axes[1].plot(sizes, curve_errors, marker="o", color="#1D4ED8")
    axes[1].set_xlabel("训练样本数")
    axes[1].set_ylabel("测试集 MAE（mm）")
    axes[1].set_title("样本增加时的误差变化")
    axes[1].grid(color="#E3E9EE", linewidth=0.8)
    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("从衍射强度曲线反演条纹间距：k-NN 教学示例", fontsize=13)
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "ml_inversion_demo.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print("数据：合成双缝衍射曲线，仅用于教学演示")
    print(f"样本数：{len(features)}，训练/测试：{len(x_train)}/{len(x_test)}")
    print(f"k-NN 邻居数：{max(1, args.neighbors)}")
    print(f"测试集 MAE：{mae:.4f} mm")
    print(f"平均相对误差：{relative:.2f}%")
    print(f"图已保存：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
