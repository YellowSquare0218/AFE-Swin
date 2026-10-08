import argparse
import csv
import json
import hashlib
from pathlib import Path
import uuid

import numpy as np
from PIL import Image
import torch

from breakhis_patient_split import patient_id_from_path
from checkpoint_io import load_checkpoint, model_from_checkpoint, test_dataset, require_trained_checkpoint
from experiment_utils import write_json
from train_afe_swin import ResizeKeepRatioPad
from frequency_statistics import (band_responses, grid_mean, geometry_mask, structure_references,
                                  spatial_spearman, amplitude_summary, summarize_pairs, frequency_masks)


def patient_mapping(paths, magnification, csv_path):
    if magnification != "Bracs":
        return {Path(path).name: patient_id_from_path(path) for path in paths}
    if csv_path is None:
        raise ValueError("BRACS needs a verified image_name,patient_id CSV. WSI or slide counts cannot be called patients.")
    mapping = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            name, patient = row["image_name"], row["patient_id"].strip()
            if not patient or name in mapping:
                raise ValueError(f"Empty or duplicate BRACS patient mapping: {name}")
            mapping[name] = patient
    missing = [Path(path).name for path in paths if Path(path).name not in mapping]
    if missing:
        raise ValueError(f"Patient map is missing {len(missing)} test images; first: {missing[0]}")
    return mapping


def fp32_band_responses(features, weight):
    tensor = torch.from_numpy(features.astype(np.float32))
    weights = torch.from_numpy(weight.astype(np.complex64))
    spectrum = torch.fft.rfft2(tensor, norm="ortho")
    result = {}
    for band, mask in frequency_masks(*tensor.shape[-2:]).items():
        selected = spectrum * torch.from_numpy(mask)
        before = torch.fft.irfft2(selected, s=tensor.shape[-2:], norm="ortho")
        after = torch.fft.irfft2(selected * weights, s=tensor.shape[-2:], norm="ortho")
        result[band] = {"before": before.square().mean(0).sqrt().numpy(),
                        "after": after.square().mean(0).sqrt().numpy()}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--patient-map", help="Verified BRACS image_name,patient_id CSV.")
    parser.add_argument("--output")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-images", type=int, help="Diagnostic prefix only; never labeled a complete test analysis.")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--examples", nargs="*", help="Preselected test filenames, not correlation-ranked examples.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--diagnostic-only", action="store_true", help="Explicitly label untrained/diagnostic exports; not formal evidence.")
    args = parser.parse_args(argv)
    if args.bootstrap <= 0:
        parser.error("--bootstrap must be positive")
    payload = load_checkpoint(args.checkpoint, args.device)
    if not args.diagnostic_only:
        require_trained_checkpoint(payload)
    with open(args.checkpoint, "rb") as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    config = payload["config"]
    if config["model_all"] not in ("AFE_Swin", "FSD_Swin", "FPE_Swin"):
        raise ValueError("Frequency analysis requires AFE-Swin, not the spatial-only baseline.")
    dataset, paths = test_dataset(payload, args.data_root)
    mapping = patient_mapping(paths, config["Magnification"], args.patient_map)
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max-images must be positive")
    count = min(args.max_images or len(dataset), len(dataset))
    names = {Path(path).name for path in paths[:count]}
    requested = set(args.examples or [Path(paths[0]).name])
    if not requested <= names:
        raise ValueError("All requested examples must be in the included test set.")
    output = Path(args.output or Path(args.checkpoint).parent / f"frequency_analysis_{uuid.uuid4().hex[:8]}")
    output.mkdir(parents=True, exist_ok=False)
    model = model_from_checkpoint(payload, args.device)
    raw_weight = model.freq_prior_module.complex_weight.detach().cpu().contiguous()
    weight = torch.view_as_complex(raw_weight).numpy()
    np.savez_compressed(output / "measured_filter.npz", weight=weight,
                        mean_absolute_deviation=np.abs(weight - 1).mean(axis=0))
    records, exclusions, examples, case_manifest = [], [], [], []
    for index in range(count):
        path = paths[index]
        name = Path(path).name
        tensor, target = dataset[index]
        with Image.open(path) as original:
            original_size = original.size
            rgb = np.asarray(ResizeKeepRatioPad(config["img_size"])(original.convert("RGB")))
        valid = geometry_mask(original_size, config["img_size"], (config["feature_hw"], config["feature_hw"]))
        if valid.sum() < 3:
            exclusions.append({"image": name, "reason": "fewer than three full-ROI feature cells"})
            continue
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device == "cuda"):
            features = model._to_bchw(model.backbone.forward_features(tensor.unsqueeze(0).to(args.device)))
        features = features[0].float().cpu().numpy()
        responses = band_responses(features, weight)
        responses32 = fp32_band_responses(features, weight)
        references = structure_references(rgb)
        reference_grid = {key: grid_mean(value, valid.shape) for key, value in references.items()}
        for band in ("low", "mid", "high"):
            for reference in ("edge", "hematoxylin"):
                before = spatial_spearman(responses[band]["before"], reference_grid[reference], valid)
                after = spatial_spearman(responses[band]["after"], reference_grid[reference], valid)
                if before is None or after is None:
                    exclusions.append({"image": name, "band": band, "reference": reference,
                                       "reason": "constant/non-finite response or reference; undefined Spearman"})
                    continue
                before32 = spatial_spearman(responses32[band]["before"], reference_grid[reference], valid)
                after32 = spatial_spearman(responses32[band]["after"], reference_grid[reference], valid)
                records.append({"image": name, "patient_id": mapping[name], "band": band, "reference": reference,
                                "before": before, "after": after, "delta": after - before,
                                "before_fp32": before32, "after_fp32": after32,
                                "valid_grid_positions": int(valid.sum()),
                                "before_amplitude": amplitude_summary(responses[band]["before"], valid),
                                "after_amplitude": amplitude_summary(responses[band]["after"], valid),
                                "enhancement_amplitude": amplitude_summary(responses[band]["enhancement"], valid)})
        if name in requested:
            examples.append({"label": f"{config['Magnification']}\n{payload['class_names'][target]}",
                             "rgb": rgb, "original_size": original_size, "responses": responses, "references": references})
            np.savez_compressed(output / f"{Path(name).stem}_measured_maps.npz", features=features, valid=valid,
                                rgb=rgb, original_size=np.asarray(original_size),
                                edge_pixels=references["edge"], hematoxylin_pixels=references["hematoxylin"],
                                edge=reference_grid["edge"], hematoxylin=reference_grid["hematoxylin"],
                                **{f"{band}_{kind}": array for band, values in responses.items() for kind, array in values.items()})
            case_manifest.append({"image": name, "class_name": payload["class_names"][target],
                                  "maps": f"{Path(name).stem}_measured_maps.npz"})
        if (index + 1) % 25 == 0:
            print(f"Analyzed {index + 1}/{count} fixed test images", flush=True)
    comparisons = []
    for band in ("low", "mid", "high"):
        for reference in ("edge", "hematoxylin"):
            selected = [row for row in records if row["band"] == band and row["reference"] == reference]
            comparisons.append(dict(summarize_pairs(selected, args.bootstrap, config["seed"]),
                                    band=band, reference=reference))
    summary = {"dataset": f"BRACS {config['task']}" if config["Magnification"] == "Bracs"
               else f"BreaKHis {config['Magnification']} {config['task']}",
               "checkpoint": str(Path(args.checkpoint).resolve()), "selected_epoch": payload["epoch"],
               "checkpoint_sha256": checkpoint_sha256, "diagnostic_only": args.diagnostic_only,
               "scope": "complete_recorded_test_set" if count == len(dataset) else "diagnostic_prefix_subset",
               "recorded_test_images": len(dataset), "included_images_before_pair_validity": count,
               "included_patient_ids_before_pair_validity": len({mapping[Path(path).name] for path in paths[:count]}),
               "patient_mapping_source": args.patient_map or "complete BreaKHis benchmark case identifier",
               "grid_mask": "cells fully inside deterministic unpadded ROI; same mask before/after; no correlation filtering",
               "bands": "r=sqrt(fx^2+fy^2)/sqrt(0.5^2+0.5^2); DC excluded; (0,1/3],(1/3,2/3],(2/3,1]",
               "response": "before=irFFT(M*FFT(F)); after=irFFT(M*FFT(F)*W); enhancement=irFFT(M*FFT(F)*(W-1)); channel RMS",
               "precision": "same EMA features; paired Fourier/RMS in float64 with an independent FP32 diagnostic",
               "comparisons": comparisons, "exclusions": exclusions,
               "clinical_segmentation_ground_truth": False}
    write_json(output / "summary.json", summary, exclusive=True)
    write_json(output / "case_manifest.json", {"checkpoint_sha256": checkpoint_sha256,
               "selected_epoch": payload["epoch"], "diagnostic_only": args.diagnostic_only,
               "selection": "preselected names or deterministic first test filename; never ranked by correlation",
               "cases": case_manifest}, exclusive=True)
    write_json(output / "per_image_pairs.json", records, exclusive=True)
    with (output / "paired_summary.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        fields = ["band", "reference", "n", "g", "before_median", "before_iqr", "after_median", "after_iqr",
                  "delta_median", "delta_iqr", "delta_ci95"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparisons)
    if not args.no_plots:
        from paper_plots import plot_filter, plot_gate, plot_cases, plot_forest
        plot_filter(weight, output / "filter_deviation")
        history = Path(args.checkpoint).parent / "history.csv"
        if history.is_file():
            plot_gate(history, payload["epoch"], output / "gate_to_validation_best")
        plot_cases(examples, output / "frequency_structure_examples")
        plot_forest([summary], output / "paired_delta_forest")
    print(f"Saved measured frequency analysis to {output}")
    return output


if __name__ == "__main__":
    main()
