from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr


def frequency_masks(height, width):
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.rfftfreq(width)[None, :]
    radius = np.sqrt(fx * fx + fy * fy) / np.sqrt(0.5 ** 2 + 0.5 ** 2)
    return {"low": (radius > 0) & (radius <= 1 / 3),
            "mid": (radius > 1 / 3) & (radius <= 2 / 3),
            "high": (radius > 2 / 3) & (radius <= 1)}


def band_responses(features, complex_weight):
    features = np.asarray(features, dtype=np.float64)
    weight = np.asarray(complex_weight, dtype=np.complex128)
    if features.ndim != 3 or weight.shape != (*features.shape[:2], features.shape[2] // 2 + 1):
        raise ValueError("Expected features CHW and complex weights C,H,W//2+1.")
    height, width = features.shape[-2:]
    spectrum = np.fft.rfft2(features, axes=(-2, -1), norm="ortho")
    result = {}
    for name, mask in frequency_masks(height, width).items():
        before = np.fft.irfft2(spectrum * mask, s=(height, width), axes=(-2, -1), norm="ortho")
        after = np.fft.irfft2(spectrum * mask * weight, s=(height, width), axes=(-2, -1), norm="ortho")
        enhancement = np.fft.irfft2(spectrum * mask * (weight - 1), s=(height, width),
                                   axes=(-2, -1), norm="ortho")
        result[name] = {"before": np.sqrt(np.mean(np.square(before), axis=0)),
                        "after": np.sqrt(np.mean(np.square(after), axis=0)),
                        "enhancement": np.sqrt(np.mean(np.square(enhancement), axis=0))}
    return result


def grid_mean(pixel_map, grid_shape=(12, 12)):
    pixel_map = np.asarray(pixel_map, dtype=np.float64)
    height, width = grid_shape
    if pixel_map.ndim != 2 or pixel_map.shape[0] % height or pixel_map.shape[1] % width:
        raise ValueError("Pixel map dimensions must be divisible by the feature grid.")
    return pixel_map.reshape(height, pixel_map.shape[0] // height,
                             width, pixel_map.shape[1] // width).mean(axis=(1, 3))


def geometry_mask(original_size, canvas_size=384, grid_shape=(12, 12)):
    width, height = original_size
    scale = canvas_size / max(width, height)
    scaled_width, scaled_height = round(width * scale), round(height * scale)
    left, top = (canvas_size - scaled_width) // 2, (canvas_size - scaled_height) // 2
    occupied = np.zeros((canvas_size, canvas_size), dtype=np.float64)
    occupied[top:top + scaled_height, left:left + scaled_width] = 1
    return grid_mean(occupied, grid_shape) == 1.0


def structure_references(rgb):

    from skimage.color import rgb2gray, rgb2hed
    from skimage.filters import sobel

    image = np.asarray(rgb, dtype=np.float64) / 255.0
    return {"edge": sobel(rgb2gray(image)), "hematoxylin": np.maximum(rgb2hed(image)[..., 0], 0)}


def spatial_spearman(response, reference, valid):
    response, reference = np.asarray(response), np.asarray(reference)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(response) & np.isfinite(reference)
    first, second = response[mask], reference[mask]
    if len(first) < 3 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return None
    return float(spearmanr(first, second).statistic)


def amplitude_summary(response, valid):
    values = np.asarray(response)[np.asarray(valid, dtype=bool)]
    return {"mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max()), "range": float(np.ptp(values))}


def summarize_pairs(rows, repeats=10000, seed=180811):
    if repeats <= 0:
        raise ValueError("Bootstrap repeats must be positive.")
    if not rows:
        return {"n": 0, "g": 0, "delta_ci95": None}
    groups = defaultdict(list)
    for row in rows:
        if not row["patient_id"]:
            raise ValueError("Bootstrap needs genuine patient_id mappings, not invented group counts.")
        if not np.isfinite([row["before"], row["after"], row["delta"]]).all():
            raise ValueError("Paired correlations must be finite.")
        groups[row["patient_id"]].append(row["delta"])
    grouped = [np.asarray(groups[patient]) for patient in sorted(groups)]
    values = {key: np.asarray([row[key] for row in rows]) for key in ("before", "after", "delta")}
    result = {"n": len(rows), "g": len(grouped), "bootstrap_unit": "patient_id",
              "statistic": "median of per-image paired delta", "bootstrap_repeats": repeats, "seed": seed}
    for key, value in values.items():
        result[f"{key}_median"] = float(np.median(value))
        result[f"{key}_iqr"] = np.quantile(value, [0.25, 0.75]).tolist()
    if len(grouped) < 2:
        result["delta_ci95"] = None
    else:
        rng = np.random.default_rng(seed)
        bootstrap = np.empty(repeats)
        for index in range(repeats):
            sampled = rng.integers(0, len(grouped), size=len(grouped))
            bootstrap[index] = np.median(np.concatenate([grouped[group] for group in sampled]))
        result["delta_ci95"] = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    return result
