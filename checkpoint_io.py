import hashlib
import json
import math
from pathlib import Path

import torch
from torchvision import datasets

from breakhis_patient_split import patient_id_from_path
from experiment_utils import PAPER_PROTOCOL
import train_afe_swin as training


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("paper_protocol") != PAPER_PROTOCOL:
        raise ValueError("Checkpoint protocol differs from the manuscript-aligned W-1 model. "
                         "Legacy bare weights cannot be relabeled as this experiment.")
    for key in ("config", "class_names", "model_ema", "epoch"):
        if key not in payload:
            raise ValueError(f"Missing checkpoint metadata: {key}")
    if "normalization" not in payload["config"] or "data_split" not in payload["config"]:
        raise ValueError("Checkpoint must retain training normalization and the original test split.")
    return payload


def model_from_checkpoint(payload, device="cpu"):
    config = payload["config"]
    model_class = getattr(training, config["model_all"])
    model = model_class(len(payload["class_names"]), pretrained=False, config=config).to(device)
    model.load_state_dict(payload["model_ema"], strict=True)
    return model.eval()


def require_trained_checkpoint(payload):
    if payload.get("diagnostic_only") or payload.get("epoch", 0) < 1:
        raise ValueError("Untrained/diagnostic checkpoint is not formal manuscript evidence")
    if not math.isfinite(payload.get("val_macro_f1", float("nan"))):
        raise ValueError("Formal visualization requires recorded validation Macro-F1")


def test_dataset(payload, data_root):
    config = payload["config"]
    _, transform = training.make_transforms(config, config["Magnification"])
    if config["Magnification"] == "Bracs":
        mapping = training.bracs_three_label if config["task"] == "three" else None
        dataset = datasets.ImageFolder(str(Path(data_root) / "test"), transform=transform, target_transform=mapping)
        if dataset.classes != training.BRACS_7_CLASS_NAMES:
            raise ValueError("BRACS test folders must preserve the official seven-class order.")
        expected = config["data_split"]["image_counts"]["test"]
        if len(dataset) != expected:
            raise ValueError("BRACS test inventory count differs from the checkpoint run.")
        if training.bracs_inventory_hash(dataset) != config["data_split"]["image_inventory_sha256"]["test"]:
            raise ValueError("BRACS test image/label inventory differs from the checkpoint run.")
        return dataset, [path for path, _ in dataset.samples]
    dataset = datasets.ImageFolder(data_root)
    if config["task"] == "eight" and dataset.classes != payload["class_names"]:
        raise ValueError("Class order differs from the checkpoint.")
    inventory = sorted((Path(path).name, int(label)) for path, label in dataset.samples)
    fingerprint = hashlib.sha256(json.dumps(inventory).encode()).hexdigest()
    if fingerprint != config["data_split"]["image_inventory_sha256"]:
        raise ValueError("BreaKHis image/label inventory differs from the checkpoint run.")
    selected_patients = set(config["data_split"]["patients"]["test"])
    selected = [index for index, (path, _) in enumerate(dataset.samples)
                if patient_id_from_path(path) in selected_patients]
    label_map = training.build_breakhis_binary_label_map(dataset.class_to_idx) if config["task"] == "binary" else None
    wrapped = training.IndexWrapper(dataset, selected, transform, label_map)
    return wrapped, [dataset.samples[index][0] for index in selected]
