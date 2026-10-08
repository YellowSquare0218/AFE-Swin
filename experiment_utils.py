import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


PAPER_PROTOCOL = "afeswin_manuscript_identity_wminus1_gap_v1"


def compute_training_normalization(samples, indices, preprocess):
    total = np.zeros(3, dtype=np.float64)
    squared = np.zeros(3, dtype=np.float64)
    pixels = 0
    if not indices:
        raise ValueError("Training normalization requires non-empty training indices.")
    for index in indices:
        with Image.open(samples[index][0]) as image:
            rgb = np.asarray(preprocess(image.convert("RGB")), dtype=np.float64) / 255.0
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Normalization expects HWC RGB images in [0, 1].")
        total += rgb.sum(axis=(0, 1))
        squared += np.square(rgb).sum(axis=(0, 1))
        pixels += rgb.shape[0] * rgb.shape[1]
    mean = total / pixels
    std = np.sqrt(np.maximum(squared / pixels - np.square(mean), 0))
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Training standard deviation must be finite and positive in every channel.")
    return {"mean": mean.tolist(), "std": std.tolist(), "images": len(indices), "pixels": pixels,
            "algorithm": "RGB [0,1]; population moments after deterministic resize/pad; no augmentation"}


def training_normalization(samples, indices, preprocess, size, cache_dir):
    files = []
    for index in indices:
        path = Path(samples[index][0])
        stat = path.stat()
        files.append([str(path.resolve()), stat.st_size, stat.st_mtime_ns])
    key = {"training_files": files, "size": size,
           "preprocess": "bilinear_antialias_white_center_pad_round_v1"}
    fingerprint = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / f"training_rgb_{fingerprint}.json"
    if path.is_file():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("fingerprint") != fingerprint or result.get("images") != len(indices):
            raise ValueError("Normalization cache does not match this training inventory.")
        if (len(result["mean"]) != 3 or len(result["std"]) != 3
                or not np.isfinite(result["mean"] + result["std"]).all()
                or any(value <= 0 for value in result["std"])):
            raise ValueError("Invalid normalization cache.")
    else:
        print(f"Computing RGB normalization on {len(indices)} TRAIN images only.", flush=True)
        result = compute_training_normalization(samples, indices, preprocess)
        result.update(fingerprint=fingerprint, input_size=size, cache_path=str(path.resolve()))
        write_json(path, result, exclusive=True)
    return result


class ValidationSelection:

    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -math.inf
        self.best_epoch = 0
        self.patience_reference = -math.inf
        self.counter = 0

    def update(self, validation_macro_f1, epoch):
        if not math.isfinite(validation_macro_f1):
            raise ValueError("Validation Macro-F1 must be finite.")
        new_best = validation_macro_f1 > self.best_score
        if new_best:
            self.best_score = validation_macro_f1
            self.best_epoch = epoch
        if validation_macro_f1 > self.patience_reference + self.min_delta:
            self.patience_reference = validation_macro_f1
            self.counter = 0
        else:
            self.counter += 1
        return new_best, self.counter >= self.patience


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False, default=str)


def source_hashes():
    root = Path(__file__).resolve().parent
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.glob("*.py"))}
