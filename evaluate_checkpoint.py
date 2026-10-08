import argparse
from pathlib import Path
import uuid

import pandas as pd
import torch
from torch.utils.data import DataLoader

from checkpoint_io import load_checkpoint, model_from_checkpoint, test_dataset
from experiment_utils import write_json
from train_afe_swin import evaluate_probs, evaluate_test_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a validation-selected EMA checkpoint; no test-based model choice.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint, args.device)
    model = model_from_checkpoint(checkpoint, args.device)
    dataset, paths = test_dataset(checkpoint, args.data_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    probabilities, labels = evaluate_probs(model, loader, torch.device(args.device), use_tta=False)
    metrics = evaluate_test_metrics(probabilities, labels, checkpoint["class_names"])
    output = Path(args.output or Path(args.checkpoint).parent / f"evaluation_{uuid.uuid4().hex[:8]}")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "metrics.json", {"checkpoint": str(Path(args.checkpoint).resolve()),
               "selected_epoch": checkpoint["epoch"], "test_images": len(dataset),
               "macro_f1_percent": metrics["macro_f1"] * 100,
               "balanced_accuracy_percent": metrics["bal_acc"] * 100,
               "mcc_percent": metrics["mcc"] * 100, "accuracy_percent": metrics["acc"]}, exclusive=True)
    pd.DataFrame(metrics["cm"], index=checkpoint["class_names"], columns=checkpoint["class_names"]).to_csv(
        output / "confusion_matrix.csv", encoding="utf-8-sig")
    pd.DataFrame({"image": paths, "target": labels, "prediction": metrics["preds"]}).to_csv(
        output / "predictions.csv", index=False, encoding="utf-8-sig")
    print(f"Saved evaluation to {output}")


if __name__ == "__main__":
    main()
