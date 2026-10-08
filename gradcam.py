import argparse
from pathlib import Path
import uuid

import numpy as np
from PIL import Image
import torch

from checkpoint_io import load_checkpoint, model_from_checkpoint, test_dataset, require_trained_checkpoint
from experiment_utils import write_json
from train_afe_swin import ResizeKeepRatioPad


def compute_gradcam(model, image, target=None):
    if image.shape[0] != 1:
        raise ValueError("Grad-CAM exports one image at a time.")
    model.eval()
    with torch.no_grad():
        raw = model.backbone.forward_features(image)
    if hasattr(model, "_to_bchw"):
        features = model._to_bchw(raw)
    elif raw.ndim == 4 and raw.shape[-1] == model.num_features:
        features = raw.permute(0, 3, 1, 2).contiguous()
    elif raw.ndim == 4 and raw.shape[1] == model.num_features:
        features = raw
    else:
        raise ValueError(f"Unsupported spatial feature shape: {raw.shape}")
    with torch.enable_grad():
        features = features.detach().float().requires_grad_(True)
        vector = model.freq_prior_module(features) if hasattr(model, "freq_prior_module") else features.mean((2, 3))
        scores = model.head(vector)
        target = int(scores.argmax(1).item()) if target is None else int(target)
        if not 0 <= target < scores.shape[1]:
            raise ValueError("Target class is outside the checkpoint class order.")
        gradients, = torch.autograd.grad(scores[0, target], features)
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * features).sum(1).relu()[0]
        maximum = heatmap.max()
        if maximum > 0:
            heatmap = heatmap / maximum
    return heatmap.detach().cpu(), scores.detach().cpu(), target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--images", nargs="+", required=True, help="Test filenames to export for this checkpoint.")
    parser.add_argument("--target", choices=("predicted", "truth"), default="predicted")
    parser.add_argument("--output")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--diagnostic-only", action="store_true")
    args = parser.parse_args(argv)
    payload = load_checkpoint(args.checkpoint, args.device)
    if not args.diagnostic_only:
        require_trained_checkpoint(payload)
    dataset, paths = test_dataset(payload, args.data_root)
    indices = {Path(path).name: index for index, path in enumerate(paths)}
    if not set(args.images) <= set(indices):
        raise ValueError("Every requested image must belong to this checkpoint's recorded test set.")
    output = Path(args.output or Path(args.checkpoint).parent / f"gradcam_{uuid.uuid4().hex[:8]}")
    output.mkdir(parents=True, exist_ok=False)
    model = model_from_checkpoint(payload, args.device)
    from matplotlib import colormaps
    from paper_plots import crop_geometry
    records = []
    for name in args.images:
        index = indices[name]
        tensor, truth = dataset[index]
        heatmap, scores, target = compute_gradcam(model, tensor.unsqueeze(0).to(args.device),
                                                target=truth if args.target == "truth" else None)
        with Image.open(paths[index]) as original:
            rgb = np.asarray(ResizeKeepRatioPad(payload["config"]["img_size"])(original.convert("RGB")))
            original_size = original.size
        size = rgb.shape[0]
        display = np.asarray(Image.fromarray(heatmap.numpy()).resize((size, size), Image.Resampling.BILINEAR))
        colored = colormaps["turbo"](display)[..., :3] * 255
        overlay = np.round(0.6 * rgb + 0.4 * colored).clip(0, 255).astype(np.uint8)
        stem = Path(name).stem
        Image.fromarray(crop_geometry(rgb, original_size)).save(output / f"{stem}_original.png")
        Image.fromarray(crop_geometry(overlay, original_size)).save(output / f"{stem}_gradcam.png")
        np.save(output / f"{stem}_feature_grid.npy", heatmap.numpy())
        records.append({"image": name, "true_class": payload["class_names"][truth],
                        "target_class": payload["class_names"][target],
                        "probabilities_fp32": torch.softmax(scores[0], 0).tolist(),
                        "nonzero_heatmap": bool(heatmap.max() > 0)})
    write_json(output / "metadata.json", {"checkpoint": str(Path(args.checkpoint).resolve()),
               "validation_selected_epoch": payload["epoch"], "method": "spatial Grad-CAM; full precision",
               "display": "bilinear enlargement only for display; original aspect ratio preserved; overlay 0.4",
               "frequency_branch_validation": False, "diagnostic_only": args.diagnostic_only, "records": records}, exclusive=True)
    print(output)
    return output


if __name__ == "__main__":
    main()
