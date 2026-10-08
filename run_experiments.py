import argparse
import json
import os
from pathlib import Path

import torch

from breakhis_patient_split import FILENAME
import train_afe_swin as training


def load_configuration(path):
    path = Path(path).resolve()
    supplied = json.loads(path.read_text(encoding="utf-8"))
    def resolve(value):
        expanded = Path(os.path.expandvars(value)).expanduser()
        return str((path.parent / expanded).resolve()) if not expanded.is_absolute() else str(expanded)

    roots = {key: resolve(value) for key, value in supplied["data_roots"].items() if value}
    output = Path(resolve(supplied["output_dir"]))
    overrides = supplied.get("training", {})
    unknown = set(overrides) - set(training.CONFIG)
    if unknown:
        raise ValueError(f"Unknown training configuration keys: {sorted(unknown)}")
    config = dict(training.CONFIG, **overrides)
    config.update(summary_csv=str(output / "paper_summary.csv"),
                  normalization_cache_dir=str(output / "normalization"),
                  breakhis_split_manifest=str(output / "splits" / f"breakhis_patient_seed{config['seed']}.json"))
    training.Data_dir.clear()
    training.Data_dir.update(roots)
    training.Save_dir.clear()
    training.Save_dir.update({key: str(output / key) for key in roots})
    training.validate_configuration(config)
    return config


def preflight_paths(tasks):
    for task in tasks:
        key = task["Magnification"]
        root = training.Data_dir.get(key)
        if root is None or not Path(root).is_dir():
            raise ValueError(f"Configure an existing data_roots['{key}'] before this run. Nothing was trained.")
        if key != "Bracs":
            expected = int(key.rstrip("X"))
            for path in Path(root).rglob("*.png"):
                match = FILENAME.fullmatch(path.name)
                if match is None or int(match.group("magnification")) != expected:
                    raise ValueError(f"Filename/magnification mismatch for {key}: {path}. Nothing was trained.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--all", action="store_true", help="Run all eight BreaKHis and two BRACS tasks.")
    parser.add_argument("--magnification", choices=("40X", "100X", "200X", "400X", "Bracs"), default="200X")
    parser.add_argument("--task", choices=("binary", "eight", "three", "seven"), default="eight")
    parser.add_argument("--model", choices=("AFE_Swin",), default="AFE_Swin")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate data, train-only moments and one loader batch; no model.")
    mode.add_argument("--smoke-test", action="store_true", help="One random-initialized real Swin forward/backward; no optimizer/train.")
    args = parser.parse_args()
    config = load_configuration(args.config)
    config["device"] = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                                    else "cpu" if args.device == "auto" else args.device)
    config["model_all"] = args.model
    tasks = training.EXPERIMENT_QUEUE if args.all else [{"Magnification": args.magnification, "task": args.task}]
    for task in tasks:
        valid = ("three", "seven") if task["Magnification"] == "Bracs" else ("binary", "eight")
        if task["task"] not in valid:
            parser.error(f"{task['Magnification']} requires task in {valid}")
    preflight_paths(tasks)
    for task in tasks:
        current = dict(config, **task)
        if not (args.dry_run or args.smoke_test):
            training.run_single_experiment(current)
            continue
        current.update(batch_size=1, num_workers=0)
        training.seed_everything(current["seed"])
        loaders = training.build_loaders(current)
        batch, target = next(iter(loaders[0]))
        report = {"task": task, "data_split": current["data_split"], "normalization": current["normalization"],
                  "batch_shape": list(batch.shape), "batch_finite": bool(torch.isfinite(batch).all()),
                  "training_launched": False, "checkpoint_written": False}
        if args.smoke_test:
            model = getattr(training, args.model)(current["num_classes"], pretrained=False, config=current)
            model = model.to(current["device"])
            logits = model(batch.to(current["device"]))
            loss = torch.nn.functional.cross_entropy(logits, target.to(current["device"]))
            loss.backward()
            report.update(model_initialization="random, diagnostic only", logits_shape=list(logits.shape),
                          loss_finite=bool(torch.isfinite(loss)), backward_completed=True)
        print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
