import os
import gc
import hashlib
import csv
import json
import random
import datetime
import warnings
import platform
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, WeightedRandomSampler

from breakhis_patient_split import split_breakhis_samples
from experiment_utils import PAPER_PROTOCOL, ValidationSelection, training_normalization, write_json, source_hashes
from afe_swin import AFE_Swin, FSD_Swin, FPE_Swin, AdaptiveFrequencyPriorBlock
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    matthews_corrcoef,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)

import timm
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2


Image.MAX_IMAGE_PIXELS = None
torch.backends.cudnn.benchmark = True


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


BRACS_7_TO_3 = {
    0: 0,
    1: 0,
    2: 0,
    3: 1,
    4: 1,
    5: 2,
    6: 2,
}

BRACS_3_CLASS_NAMES = ["Benign", "Atypia", "Malignant"]
BRACS_7_CLASS_NAMES = ["0_N", "1_PB", "2_UDH", "3_FEA", "4_ADH", "5_DCIS", "6_IC"]


def bracs_three_label(label):
    return BRACS_7_TO_3[label]

Data_dir = {}
Save_dir = {}


CONFIG = {
    "Magnification": "100X",
    "task": "eight",
    "summary_csv": "outputs/summary.csv",
    "normalization_cache_dir": "outputs/normalization",

    "model_name": "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
    "model_all": "AFE_Swin",
    "pretrained": True,
    "img_size": 384,
    "feature_hw": 12,

    "batch_size": 16,
    "accum_iter": 1,
    "epochs": 100,
    "seed": 180811,
    "breakhis_split_manifest": None,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    "backbone_lr": 1e-5,
    "head_lr": 1e-4,
    "weight_decay": 0.05,

    "early_stop_patience": 15,
    "early_stop_min_delta": 1e-4,

    "mixup": 0.4,
    "cutmix": 1.0,
    "label_smoothing": 0.05,

    "model_ema_decay": 0.999,
    "use_tta": False,
    "num_workers": 8,
    "head_dropout": 0.2,
    "rotation_breakhis": 15,
    "rotation_bracs": 10,
    "horizontal_flip_probability": 0.5,
    "vertical_flip_probability": 0.5,
    "jitter_brightness": 0.2,
    "jitter_contrast": 0.2,
    "jitter_saturation": 0.1,
    "mixup_probability": 1.0,
    "cutmix_switch_probability": 0.7,
    "cosine_min_lr": 1e-6,
    "gradient_clip_norm": 5.0,
    "exclude_bias_norm_from_weight_decay": True,
}

EXPERIMENT_QUEUE = [
    {"Magnification": "40X", "task": "binary"},
    {"Magnification": "100X", "task": "binary"},
    {"Magnification": "200X", "task": "binary"},
    {"Magnification": "400X", "task": "binary"},
    {"Magnification": "40X", "task": "eight"},
    {"Magnification": "100X", "task": "eight"},
    {"Magnification": "200X", "task": "eight"},
    {"Magnification": "400X", "task": "eight"},
    {"Magnification": "Bracs", "task": "three"},
    {"Magnification": "Bracs", "task": "seven"},
]


class ResizeKeepRatioPad:

    def __init__(self, size: int, fill=(255, 255, 255)):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = TF.resize(img, (new_h, new_w), antialias=True)
        pad_w = self.size - new_w
        pad_h = self.size - new_h
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        img = TF.pad(img, [left, top, right, bottom], fill=self.fill)
        return img


class IndexWrapper(torch.utils.data.Dataset):

    def __init__(self, ds, idx, transform, label_map=None):
        self.ds = ds
        self.idx = idx
        self.transform = transform
        self.label_map = label_map

    def __getitem__(self, i):
        orig_idx = self.idx[i]
        x, y = self.ds[orig_idx]
        if self.label_map is not None:
            y = self.label_map[y]
        return self.transform(x), y

    def __len__(self):
        return len(self.idx)


def build_breakhis_binary_label_map(class_to_idx):

    benign_keywords = [
        "adenosis", "fibroadenoma", "phyllodes", "tubular",
        "sob_a", "sob_f", "sob_pt", "sob_ta",
    ]
    benign_short = {"A", "F", "PT", "TA"}

    label_map = {}
    for class_name, orig_idx in class_to_idx.items():
        name_lower = class_name.lower()
        is_benign = class_name in benign_short or any(k in name_lower for k in benign_keywords)
        label_map[orig_idx] = 0 if is_benign else 1

    print("BreaKHis binary label mapping:")
    for class_name, orig_idx in class_to_idx.items():
        print(f"  {class_name:30s} -> {label_map[orig_idx]}")
    return label_map


def make_transforms(config, mean_key):
    mean = config["normalization"]["mean"]
    std = config["normalization"]["std"]
    setting = dict(CONFIG, **config)
    train_tf = transforms.Compose([
        ResizeKeepRatioPad(config["img_size"]),
        transforms.RandomHorizontalFlip(setting["horizontal_flip_probability"]),
        transforms.RandomVerticalFlip(setting["vertical_flip_probability"]),
        transforms.RandomRotation(setting["rotation_bracs" if mean_key == "Bracs" else "rotation_breakhis"]),
        transforms.ColorJitter(brightness=setting["jitter_brightness"], contrast=setting["jitter_contrast"],
                               saturation=setting["jitter_saturation"]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_tf = transforms.Compose([
        ResizeKeepRatioPad(config["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


SUMMARY_FIELDS = [
    "Magnification", "Timestamp", "task", "model_name", "Model_All",
    "batch_size", "Backbone_LR", "head_lr", "Best_Val_F1", "Best_Val_Acc",
    "Test_Acc", "Test_F1_Macro", "Test_F1_Weighted", "Test_Balanced_Acc",
    "Test_MCC", "Test_Kappa", "Mixup", "CutMix", "Exp_Dir",
]


def save_summary(config, results, timestamp, exp_dir):
    csv_path = config["summary_csv"]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    row_data = {
        "Magnification": config["Magnification"],
        "Timestamp": timestamp,
        "task": config.get("task", "N/A"),
        "model_name": config["model_name"],
        "Model_All": config["model_all"],
        "batch_size": config["batch_size"],
        "Backbone_LR": config["backbone_lr"],
        "head_lr": config["head_lr"],
        "Best_Val_F1": f"{results.get('best_val_f1', 0.0):.4f}",
        "Best_Val_Acc": f"{results.get('best_val_acc', 0.0):.2f}%",
        "Test_Acc": f"{results['test_acc']:.2f}%",
        "Test_F1_Macro": f"{results['test_f1_macro']:.4f}",
        "Test_F1_Weighted": f"{results['test_f1_weighted']:.4f}",
        "Test_Balanced_Acc": f"{results['test_bal_acc']:.4f}",
        "Test_MCC": f"{results['test_mcc']:.4f}",
        "Test_Kappa": f"{results['test_kappa']:.4f}",
        "Mixup": config.get("mixup", 0.0),
        "CutMix": config.get("cutmix", 0.0),
        "Exp_Dir": exp_dir,
    }
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)


def get_parameter_groups(model, config):
    skip = {}
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if config["exclude_bias_norm_from_weight_decay"] and (param.ndim <= 1 or name.endswith(".bias") or name in skip):
            this_wd = 0.0
        else:
            this_wd = config["weight_decay"]

        if "backbone" in name:
            backbone_params.append({"params": param, "lr": config["backbone_lr"], "weight_decay": this_wd})
        else:
            head_params.append({"params": param, "lr": config["head_lr"], "weight_decay": this_wd})

    return backbone_params + head_params


def evaluate_probs(model, loader, device, use_tta=False):
    model.eval()
    all_probs, all_labels = [], []
    with torch.inference_mode():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            views = [imgs]
            if use_tta:
                views.append(torch.flip(imgs, dims=[3]))
                views.append(torch.flip(imgs, dims=[2]))

            probs_sum = None
            for view in views:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = model(view)
                    probs = F.softmax(logits, dim=1)
                probs_sum = probs if probs_sum is None else probs_sum + probs

            probs_mean = probs_sum / len(views)
            all_probs.append(probs_mean.cpu())
            all_labels.extend(labels.cpu().numpy())

    return torch.cat(all_probs), np.array(all_labels)


def evaluate_test_metrics(probs, labels, class_names):
    labels = np.array(labels)
    preds = probs.argmax(dim=1).cpu().numpy()
    acc = accuracy_score(labels, preds) * 100
    macro_f1 = f1_score(labels, preds, labels=list(range(len(class_names))), average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted")
    bal_acc = recall_score(labels, preds, labels=list(range(len(class_names))), average="macro", zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    report_dict = classification_report(
        labels, preds, labels=list(range(len(class_names))), target_names=class_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "bal_acc": bal_acc,
        "mcc": mcc,
        "kappa": kappa,
        "report_dict": report_dict,
        "cm": cm,
        "preds": preds,
    }


def _dataloader_kwargs(config, shuffle=False, sampler=None, drop_last=False):
    kwargs = dict(
        batch_size=config["batch_size"],
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=config["num_workers"],
        drop_last=drop_last,
        pin_memory=True,
    )
    if config["num_workers"] > 0:
        kwargs.update(dict(persistent_workers=True, prefetch_factor=4))
    return kwargs


def bracs_inventory_hash(dataset):
    inventory = sorted((Path(path).relative_to(dataset.root).as_posix(), int(label))
                       for path, label in dataset.samples)
    return hashlib.sha256(json.dumps(inventory).encode()).hexdigest()


def build_loaders(config):
    mean_key = config["Magnification"]
    if mean_key not in Data_dir or not Data_dir[mean_key]:
        raise ValueError(f"Configure a valid data_roots['{mean_key}'] before loading data.")

    if config["Magnification"] == "Bracs":
        data_root = Data_dir["Bracs"]
        if config["task"] == "three":
            target_tf = bracs_three_label
            class_names = BRACS_3_CLASS_NAMES
            config["num_classes"] = 3
        elif config["task"] == "seven":
            target_tf = None
            class_names = BRACS_7_CLASS_NAMES
            config["num_classes"] = 7
        else:
            raise ValueError("BRACS task must be 'three' or 'seven'.")

        train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), target_transform=target_tf)
        val_ds = datasets.ImageFolder(os.path.join(data_root, "val"), target_transform=target_tf)
        test_ds = datasets.ImageFolder(os.path.join(data_root, "test"), target_transform=target_tf)
        assert train_ds.class_to_idx == val_ds.class_to_idx == test_ds.class_to_idx
        if train_ds.classes != BRACS_7_CLASS_NAMES:
            raise ValueError("BRACS folders must preserve the official 0_N ... 6_IC label order.")
        config["normalization"] = training_normalization(
            train_ds.samples, list(range(len(train_ds))), ResizeKeepRatioPad(config["img_size"]),
            config["img_size"], config["normalization_cache_dir"],
        )
        train_tf, eval_tf = make_transforms(config, mean_key)
        train_ds.transform, val_ds.transform, test_ds.transform = train_tf, eval_tf, eval_tf
        config["data_split"] = {"protocol": "bracs_official_train_val_test_v1",
                                "image_counts": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
                                "image_inventory_sha256": {name: bracs_inventory_hash(ds) for name, ds in
                                                           (("train", train_ds), ("val", val_ds), ("test", test_ds))}}

        train_dl = DataLoader(train_ds, **_dataloader_kwargs(config, shuffle=True, drop_last=True))
        val_dl = DataLoader(val_ds, **_dataloader_kwargs(config, shuffle=False))
        test_dl = DataLoader(test_ds, **_dataloader_kwargs(config, shuffle=False))
        return train_dl, val_dl, test_dl, class_names

    data_root = Data_dir[config["Magnification"]]
    dataset = datasets.ImageFolder(data_root, transform=None)
    if len(dataset.classes) != 8:
        raise ValueError("BreaKHis input requires the eight original subtype folders for both tasks.")

    if config["task"] == "binary":
        label_map = build_breakhis_binary_label_map(dataset.class_to_idx)
        targets = [label_map[t] for t in dataset.targets]
        class_names = ["Benign", "Malignant"]
        config["num_classes"] = 2
    elif config["task"] == "eight":
        label_map = None
        targets = dataset.targets
        class_names = dataset.classes
        config["num_classes"] = 8
    else:
        raise ValueError("BreaKHis task must be 'binary' or 'eight'.")

    manifest_path = config.get("breakhis_split_manifest") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "splits",
        f"breakhis_patient_seed{config['seed']}.json",
    )
    split_indices, split_metadata = split_breakhis_samples(
        dataset.samples,
        [Data_dir[key] for key in ("40X", "100X", "200X", "400X") if Data_dir.get(key)],
        config["seed"], manifest_path,
        expected_magnification=int(config["Magnification"].rstrip("X")),
    )
    train_idx, val_idx, test_idx = (split_indices[name] for name in ("train", "val", "test"))
    config["breakhis_data_split"] = split_metadata
    config["data_split"] = split_metadata
    print(f"BreaKHis shared patient counts: {split_metadata['patient_counts']}")
    print(f"Current magnification patient counts: {split_metadata['current_patient_counts']}")
    print(f"Current magnification image counts: {split_metadata['image_counts']}")
    config["normalization"] = training_normalization(
        dataset.samples, train_idx, ResizeKeepRatioPad(config["img_size"]),
        config["img_size"], config["normalization_cache_dir"],
    )
    train_tf, eval_tf = make_transforms(config, mean_key)

    train_targets_list = [targets[i] for i in train_idx]
    class_counts = np.bincount(train_targets_list, minlength=config["num_classes"]).astype(np.float32)
    if np.any(class_counts == 0):
        raise ValueError("Patient-level training split is missing a target class at this magnification. "
                         "Check the dataset and recorded patient inventory; do not fall back to image splitting.")
    class_counts[class_counts == 0] = 1.0
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in train_targets_list]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_dl = DataLoader(
        IndexWrapper(dataset, train_idx, train_tf, label_map),
        **_dataloader_kwargs(config, sampler=sampler, drop_last=True),
    )
    val_dl = DataLoader(
        IndexWrapper(dataset, val_idx, eval_tf, label_map),
        **_dataloader_kwargs(config, shuffle=False),
    )
    test_dl = DataLoader(
        IndexWrapper(dataset, test_idx, eval_tf, label_map),
        **_dataloader_kwargs(config, shuffle=False),
    )
    return train_dl, val_dl, test_dl, class_names


def validate_configuration(config):
    if config["use_tta"]:
        raise ValueError("TTA is not part of the manuscript evaluation protocol")
    if config["accum_iter"] != 1:
        raise ValueError("The manuscript uses batch_size=16 without gradient accumulation")
    for key in ("batch_size", "epochs", "early_stop_patience", "img_size", "feature_hw"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")


def run_single_experiment(params):
    config = CONFIG.copy()
    config.update(params)
    validate_configuration(config)
    seed_everything(config["seed"], deterministic=True)
    device = config["device"]
    use_amp = device.type == "cuda"
    train_dl, val_dl, test_dl, class_names = build_loaders(config)
    if len(train_dl) == 0:
        raise ValueError("Training split has fewer images than batch_size with drop_last=True.")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "patient622" if "breakhis_data_split" in config else "official"
    exp_dir = os.path.join(Save_dir[config["Magnification"]],
                           f"{timestamp}_{config['model_all']}_{config['task']}_{suffix}_{uuid.uuid4().hex[:8]}")
    os.makedirs(exp_dir, exist_ok=False)
    config["paper_protocol"] = PAPER_PROTOCOL
    snapshot = json.loads(json.dumps(config, default=str))
    environment = {"python": platform.python_version(), "torch": str(torch.__version__),
                   "timm": timm.__version__, "device": str(device),
                   "manuscript_torch_version": "2.5.1", "manuscript_python_version": "3.11.14",
                   "package_revision": "20261008_v2", "source_sha256": source_hashes()}
    environment["matches_manuscript_torch"] = str(torch.__version__).split("+")[0] == "2.5.1"
    environment["matches_manuscript_python"] = platform.python_version() == "3.11.14"
    write_json(os.path.join(exp_dir, "config.json"), snapshot, exclusive=True)
    write_json(os.path.join(exp_dir, "environment.json"), environment, exclusive=True)
    write_json(os.path.join(exp_dir, "data_split.json"), config["data_split"], exclusive=True)
    write_json(os.path.join(exp_dir, "normalization.json"), config["normalization"], exclusive=True)
    print(f"Run: {exp_dir}\nClasses: {class_names}", flush=True)
    if config["model_all"] in ("AFE_Swin", "FSD_Swin", "FPE_Swin"):
        print("Literal manuscript model: GAP removes non-DC classification information. "
              "The initial classification gradients of W and alpha are zero.")
    use_mixup = config["mixup"] > 0.0 or config["cutmix"] > 0.0
    if use_mixup:
        mixup_fn = Mixup(
            mixup_alpha=config["mixup"], cutmix_alpha=config["cutmix"],
            prob=config["mixup_probability"], switch_prob=config["cutmix_switch_probability"], mode="batch",
            label_smoothing=config["label_smoothing"], num_classes=config["num_classes"],
        )
        criterion_train = SoftTargetCrossEntropy()
    else:
        mixup_fn = None
        criterion_train = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    model_class = globals()[config["model_all"]]
    model = model_class(num_classes=config["num_classes"], pretrained=config["pretrained"], config=config).to(device)
    model = model.to(memory_format=torch.channels_last)
    model_ema = ModelEmaV2(model, decay=config["model_ema_decay"], device=device)
    optimizer = optim.AdamW(get_parameter_groups(model, config))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["cosine_min_lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    selection = ValidationSelection(config["early_stop_patience"], config["early_stop_min_delta"])
    best_path = None
    best_val_acc = None
    history_fields = ["epoch", "train_loss", "val_macro_f1", "val_accuracy_pct", "raw_alpha", "ema_alpha",
                      "backbone_lr", "head_lr", "new_validation_best", "patience_counter"]
    def gate_value(instance):
        block = getattr(instance, "freq_prior_module", None)
        return None if block is None else block.alpha.detach().cpu().item()

    if hasattr(model, "freq_prior_module"):
        weight = model.freq_prior_module.complex_weight.detach().cpu().contiguous()
        np.savez_compressed(os.path.join(exp_dir, "initial_frequency.npz"),
                            complex_weight=torch.view_as_complex(weight).numpy(), alpha=np.array(0.0))
    with open(os.path.join(exp_dir, "history.csv"), "x", encoding="utf-8", newline="") as history:
        writer = csv.DictWriter(history, fieldnames=history_fields)
        writer.writeheader()
        writer.writerow({"epoch": 0, "raw_alpha": gate_value(model), "ema_alpha": gate_value(model_ema.module)})
        history.flush()
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            for step, (x, y) in enumerate(train_dl):
                x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                y = y.to(device, non_blocking=True)
                if mixup_fn is not None:
                    x, y = mixup_fn(x, y)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    logits = model(x)
                    loss = criterion_train(logits, y) / config["accum_iter"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss; no metrics are fabricated.")
                scaler.scale(loss).backward()
                if ((step + 1) % config["accum_iter"] == 0) or ((step + 1) == len(train_dl)):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["gradient_clip_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                    model_ema.update(model)
                    optimizer.zero_grad(set_to_none=True)
                running_loss += loss.item() * config["accum_iter"]
            scheduler.step()
            val_probs, val_labels = evaluate_probs(model_ema.module, val_dl, device, use_tta=False)
            val_metrics = evaluate_test_metrics(val_probs, val_labels, class_names)
            val_f1, val_acc = val_metrics["macro_f1"], val_metrics["acc"]
            new_best, stop = selection.update(val_f1, epoch)
            if new_best:
                best_path = os.path.join(exp_dir, f"best_ema_epoch_{epoch:03d}.pt")
                checkpoint = {"paper_protocol": PAPER_PROTOCOL, "epoch": epoch, "val_macro_f1": val_f1,
                              "val_accuracy_pct": val_acc, "config": snapshot, "class_names": class_names,
                              "environment": environment,
                              "model_ema": {key: value.detach().cpu() for key, value in model_ema.module.state_dict().items()}}
                with open(best_path, "xb") as stream:
                    torch.save(checkpoint, stream)
                best_val_acc = val_acc
                write_json(os.path.join(exp_dir, "best_checkpoint.json"),
                           {"path": os.path.basename(best_path), "epoch": epoch, "validation_macro_f1": val_f1,
                            "selection": "EMA validation Macro-F1 only; ties keep the earlier epoch"})
            writer.writerow({"epoch": epoch, "train_loss": running_loss / len(train_dl),
                             "val_macro_f1": val_f1, "val_accuracy_pct": val_acc,
                             "raw_alpha": gate_value(model), "ema_alpha": gate_value(model_ema.module),
                             "backbone_lr": optimizer.param_groups[0]["lr"], "head_lr": optimizer.param_groups[-1]["lr"],
                             "new_validation_best": new_best, "patience_counter": selection.counter})
            history.flush()
            print(f"Epoch {epoch}/{config['epochs']} | Val EMA Macro-F1 {val_f1:.6f} | "
                  f"Patience {selection.counter}/{selection.patience}", flush=True)
            if stop:
                break
    if best_path is None:
        raise RuntimeError("No validation-selected checkpoint was produced.")
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_ema"], strict=True)
    probs, labels = evaluate_probs(model, test_dl, device, use_tta=False)
    metrics = evaluate_test_metrics(probs, labels, class_names)
    pd.DataFrame(metrics["report_dict"]).T.to_csv(os.path.join(exp_dir, "classification_report.csv"), encoding="utf-8-sig")
    pd.DataFrame(metrics["cm"], index=class_names, columns=class_names).to_csv(
        os.path.join(exp_dir, "confusion_matrix.csv"), encoding="utf-8-sig")
    if isinstance(test_dl.dataset, IndexWrapper):
        test_paths = [test_dl.dataset.ds.samples[index][0] for index in test_dl.dataset.idx]
    else:
        test_paths = [sample[0] for sample in test_dl.dataset.samples]
    pd.DataFrame({"image_path": test_paths, "target": labels, "prediction": metrics["preds"],
                  **{f"probability_{name}": probs[:, index].numpy() for index, name in enumerate(class_names)}}).to_csv(
        os.path.join(exp_dir, "test_predictions.csv"), index=False)
    test_raw = {"accuracy": metrics["acc"] / 100, "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["bal_acc"], "mcc": metrics["mcc"]}
    write_json(os.path.join(exp_dir, "result.json"),
               {"paper_protocol": PAPER_PROTOCOL, "best_epoch": selection.best_epoch, "trained_epochs": epoch,
                "checkpoint": os.path.basename(best_path), "best_validation_macro_f1": selection.best_score,
                "test": test_raw, "test_percent": {key: value * 100 for key, value in test_raw.items()},
                "test_images": len(labels), "test_evaluations": 1,
                "old_manuscript_metrics_reproduced": False}, exclusive=True)
    save_summary(config, {"best_val_f1": selection.best_score, "best_val_acc": best_val_acc,
                          "test_acc": metrics["acc"], "test_f1_macro": metrics["macro_f1"],
                          "test_f1_weighted": metrics["weighted_f1"], "test_bal_acc": metrics["bal_acc"],
                          "test_mcc": metrics["mcc"], "test_kappa": metrics["kappa"]}, timestamp, exp_dir)
    del model, model_ema, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Finished: {exp_dir}. Test evaluated once using validation-selected EMA epoch {selection.best_epoch}.")
    return exp_dir


def main():
    from run_experiments import main as run_cli
    run_cli()


if __name__ == "__main__":
    main()
