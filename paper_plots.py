import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import PowerNorm
import numpy as np
from PIL import Image


if os.environ.get("AFE_FONT_DIR"):
    for font in Path(os.environ["AFE_FONT_DIR"]).glob("*"):
        if font.suffix.lower() in (".ttf", ".ttc", ".otf"):
            font_manager.fontManager.addfont(str(font))
cjk_candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"]
available_fonts = {font.name for font in font_manager.fontManager.ttflist}
cjk_font = next((name for name in cjk_candidates if name in available_fonts), "DejaVu Sans")
plt.rcParams.update({"font.family": ["Arial" if "Arial" in available_fonts else "DejaVu Sans", cjk_font],
                     "font.sans-serif": ["Arial", *cjk_candidates, "DejaVu Sans"],
                     "mathtext.fontset": "custom", "mathtext.rm": cjk_font, "mathtext.sf": cjk_font,
                     "mathtext.it": "Arial:italic" if "Arial" in available_fonts else "DejaVu Sans:italic",
                     "mathtext.bf": "Arial:bold" if "Arial" in available_fonts else "DejaVu Sans:bold",
                     "mathtext.cal": "DejaVu Sans",
                     "font.size": 11, "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42})


def save_figure(figure, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(stem.with_suffix("." + extension), dpi=400, bbox_inches="tight")
    plt.close(figure)


def plot_filter(weight, output):
    magnitude = np.abs(weight - 1).mean(axis=0)
    height, width_half = magnitude.shape
    fx, fy = np.fft.rfftfreq(2 * (width_half - 1)), np.fft.fftshift(np.fft.fftfreq(height))
    figure, axis = plt.subplots(figsize=(5.3, 4.6))
    maximum = float(magnitude.max())
    image = axis.pcolormesh(fx, fy, np.fft.fftshift(magnitude, axes=0), shading="nearest", cmap="viridis",
                           vmin=0, vmax=maximum if maximum > 0 else 1)
    axis.set_xlabel(r"水平特征网格频率 $f_x$（周期/特征单元）", fontfamily=cjk_font)
    axis.set_ylabel(r"垂直特征网格频率 $f_y$（周期/特征单元）", fontfamily=cjk_font)
    figure.colorbar(image, ax=axis, pad=0.03).set_label(r"通道平均 $|W_c-1|$", fontfamily=cjk_font)
    save_figure(figure, output)


def plot_gate(history_path, selected_epoch, output):
    with open(history_path, encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["epoch"]) <= selected_epoch]
    if not rows or any(not row["raw_alpha"] or not row["ema_alpha"] for row in rows):
        raise ValueError("Real Raw/EMA alpha history is required; no curve is reconstructed from a screenshot.")
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(selected_epoch + 1)):
        raise ValueError("History must contain every epoch from initialization through the selected epoch")
    if not np.isfinite([[float(row["raw_alpha"]), float(row["ema_alpha"])] for row in rows]).all():
        raise ValueError("Alpha history must be finite")
    if float(rows[0]["raw_alpha"]) != 0 or float(rows[0]["ema_alpha"]) != 0:
        raise ValueError("Manuscript gate must start at zero initialization")
    figure, axis = plt.subplots(figsize=(5.2, 4.5))
    axis.plot(epochs, [float(row["raw_alpha"]) for row in rows], color="#355e7c", label=r"Raw $\alpha$")
    axis.plot(epochs, [float(row["ema_alpha"]) for row in rows], color="#b22235", label=r"EMA $\alpha$")
    axis.axvline(selected_epoch, linestyle=":", color="#666666", label=f"验证集Macro-F1最优：Epoch {selected_epoch}")
    axis.set_xlabel("训练轮次")
    axis.set_ylabel(r"门控系数 $\alpha$", fontfamily=cjk_font)
    axis.grid(alpha=0.15)
    axis.legend(fontsize=9, loc="best")
    save_figure(figure, output)


def crop_geometry(array, original_size):
    width, height = original_size
    size = array.shape[0]
    scale = size / max(width, height)
    new_width, new_height = round(width * scale), round(height * scale)
    left, top = (size - new_width) // 2, (size - new_height) // 2
    return array[top:top + new_height, left:left + new_width]


def plot_cases(examples, output):
    if not examples:
        return
    figure, axes = plt.subplots(len(examples), 6, figsize=(12, 1.7 * len(examples) + 0.7), squeeze=False)
    titles = ["原图", "低频增强响应", "中频增强响应", "高频增强响应", "边缘强度", "H染色密度"]
    response_max = max(float(example["responses"][band]["enhancement"].max())
                       for example in examples for band in ("low", "mid", "high"))
    response_norm = PowerNorm(0.5, vmin=0, vmax=response_max if response_max > 0 else 1)
    edge_max = max(float(example["references"]["edge"].max()) for example in examples)
    stain_max = max(float(example["references"]["hematoxylin"].max()) for example in examples)
    for row, example in enumerate(examples):
        axes[row, 0].imshow(crop_geometry(example["rgb"], example["original_size"]))
        axes[row, 0].set_ylabel(example["label"], fontsize=10)
        for column, band in enumerate(("low", "mid", "high"), start=1):
            coarse = example["responses"][band]["enhancement"].astype(np.float32)

            size = example["rgb"].shape[0]
            display = np.asarray(Image.fromarray(coarse).resize((size, size), Image.Resampling.NEAREST))
            response_image = axes[row, column].imshow(crop_geometry(display, example["original_size"]),
                                                       norm=response_norm, cmap="magma")
        edge_image = axes[row, 4].imshow(crop_geometry(example["references"]["edge"], example["original_size"]),
                                         cmap="gray", vmin=0, vmax=edge_max if edge_max > 0 else 1)
        stain_image = axes[row, 5].imshow(crop_geometry(example["references"]["hematoxylin"], example["original_size"]),
                                          cmap="gray", vmin=0, vmax=stain_max if stain_max > 0 else 1)
        for column, axis in enumerate(axes[row]):
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(titles[column], fontsize=11)
    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.22, top=0.90, wspace=0.07, hspace=0.20)

    figure.canvas.draw()
    positions = [axis.get_position() for axis in axes[-1]]
    bar_y = positions[1].y0 - 0.045
    inset = 0.015
    response_bar = figure.add_axes([positions[1].x0 + inset, bar_y,
                                   positions[3].x1 - positions[1].x0 - 2 * inset, 0.014])
    edge_bar = figure.add_axes([positions[4].x0 + inset, bar_y, positions[4].width - 2 * inset, 0.014])
    stain_bar = figure.add_axes([positions[5].x0 + inset, bar_y, positions[5].width - 2 * inset, 0.014])
    figure.colorbar(response_image, cax=response_bar, orientation="horizontal").set_label("通道RMS（平方根色阶）")
    figure.colorbar(edge_image, cax=edge_bar, orientation="horizontal").set_label("边缘强度")
    figure.colorbar(stain_image, cax=stain_bar, orientation="horizontal").set_label("H染色密度")
    save_figure(figure, output)


def plot_forest(summaries, output):
    figure, axes = plt.subplots(len(summaries), 1, figsize=(8.2, 3.3 * len(summaries)), squeeze=False, sharex=True)
    names = {"low": "低频", "mid": "中频", "high": "高频", "edge": "边缘强度", "hematoxylin": "苏木精染色密度"}
    for block, summary in enumerate(summaries):
        axis = axes[block, 0]
        for row, comparison in enumerate(summary["comparisons"]):
            if "delta_median" in comparison:
                median = comparison["delta_median"]
                if comparison["delta_ci95"] is not None:
                    low, high = comparison["delta_ci95"]
                    axis.plot([low, high], [row, row], color="#355e7c", linewidth=1.8)
                axis.scatter([median], [row], s=32, color="#b22235", zorder=3)
            axis.text(1.03, row, f"{comparison['n']}/{comparison['g']}", transform=axis.get_yaxis_transform(),
                      va="center", fontsize=10)
        axis.set_yticks(range(len(summary["comparisons"])),
                        [f"{names[row['band']]} - {names[row['reference']]}" for row in summary["comparisons"]])
        axis.invert_yaxis()
        axis.axvline(0, color="#777777", linestyle="--", linewidth=1)
        axis.set_title(summary["dataset"], loc="left", fontsize=11)
        axis.text(1.03, 1.02, "n/g", transform=axis.transAxes, fontsize=10)
        axis.grid(axis="x", alpha=0.15)
    axes[-1, 0].set_xlabel(r"配对Spearman相关系数变化 $\Delta\rho$（中位数及患者成组95%CI）", fontfamily=cjk_font)
    figure.subplots_adjust(left=0.31, right=0.88, hspace=0.3)
    save_figure(figure, output)
