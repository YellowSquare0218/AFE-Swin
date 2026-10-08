import argparse
import csv
import json
from pathlib import Path
import uuid

import numpy as np

from analyze_frequency import main as analyze_main, patient_mapping
from checkpoint_io import load_checkpoint, test_dataset, require_trained_checkpoint
from experiment_utils import write_json
from gradcam import main as gradcam_main
from paper_plots import plot_filter, plot_gate, plot_cases, plot_forest


def validate_training_history(path, payload):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or [int(row["epoch"]) for row in rows] != list(range(len(rows))):
        raise ValueError("Real history must include initialization and all consecutive training epochs")
    if float(rows[0]["raw_alpha"]) != 0 or float(rows[0]["ema_alpha"]) != 0:
        raise ValueError("The manuscript specifies zero gate initialization")
    scored = rows[1:]
    if not scored or not np.isfinite([float(row["val_macro_f1"]) for row in scored]).all():
        raise ValueError("Complete validation Macro-F1 history is required")
    best = max(scored, key=lambda row: float(row["val_macro_f1"]))
    if int(best["epoch"]) != payload["epoch"] or not np.isclose(
            float(best["val_macro_f1"]), payload["val_macro_f1"], rtol=0, atol=1e-12):
        raise ValueError("Checkpoint does not match the final validation-selected EMA model in this history")
    alpha = payload.get("model_ema", {}).get("freq_prior_module.alpha")
    if alpha is not None and not np.isclose(float(alpha.item()), float(best["ema_alpha"]), rtol=0, atol=1e-12):
        raise ValueError("Logged EMA alpha does not match the selected checkpoint tensor")


def select_cases(paths, labels, class_names, selections):
    if len(paths) != len(labels) or len({Path(path).name for path in paths}) != len(paths):
        raise ValueError("Test paths must have unique filenames and paired class labels")
    selected = []
    for selection in selections:
        class_name = selection["class_name"]
        if class_name not in class_names:
            raise ValueError(f"Requested class {class_name} not in checkpoint classes: {class_names}")
        index = class_names.index(class_name)
        candidates = sorted(Path(path).name for path, label in zip(paths, labels) if label == index)
        name = selection.get("image_name") or (candidates[0] if candidates else None)
        if name not in candidates:
            raise ValueError(f"Requested image {name} is absent from recorded test class {class_name}")
        selected.append(name)
    if len(set(selected)) != len(selected):
        raise ValueError("Requested cases must be distinct")
    return selected


def load_measured_case(path, label):
    with np.load(path, allow_pickle=False) as maps:
        return {"rgb": maps["rgb"], "original_size": tuple(int(value) for value in maps["original_size"]),
                "label": label,
                "references": {"edge": maps["edge_pixels"], "hematoxylin": maps["hematoxylin_pixels"]},
                "responses": {band: {kind: maps[f"{band}_{kind}"] for kind in ("before", "after", "enhancement")}
                              for band in ("low", "mid", "high")}}


def write_supplementary_table(summaries, path):
    fields = ["dataset", "band", "reference", "n", "g", "before_median", "before_q1", "before_q3",
              "after_median", "after_q1", "after_q3", "delta_median", "delta_q1", "delta_q3",
              "delta_ci95_low", "delta_ci95_high"]
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            for comparison in summary["comparisons"]:
                row = {key: comparison[key] for key in ("band", "reference", "n", "g")}
                row["dataset"] = summary["dataset"]
                for kind in ("before", "after", "delta"):
                    row[f"{kind}_median"] = comparison.get(f"{kind}_median")
                    row[f"{kind}_q1"], row[f"{kind}_q3"] = comparison.get(f"{kind}_iqr", [None, None])
                row["delta_ci95_low"], row["delta_ci95_high"] = comparison["delta_ci95"] or [None, None]
                writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gradcam", action="store_true", help="Also export AFE-only spatial Grad-CAM for the six Fig.6 diagnoses.")
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    def resolve(value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else (config_path.parent / path).resolve()

    expected = {"breakhis_200x": ("200X", "eight"), "bracs_seven": ("Bracs", "seven")}
    if set(config["datasets"]) != set(expected):
        raise ValueError("The paper protocol requires BreaKHis 200X eight-class and BRACS seven-class")
    jobs = {}
    for key, task in expected.items():
        supplied = config["datasets"][key]
        checkpoint, data_root = resolve(supplied["checkpoint"]), resolve(supplied["data_root"])
        payload = load_checkpoint(checkpoint)
        require_trained_checkpoint(payload)
        if (payload["config"]["Magnification"], payload["config"]["task"]) != task:
            raise ValueError(f"Checkpoint task mismatch: {key}")
        if payload["config"]["model_all"] not in ("AFE_Swin", "FSD_Swin", "FPE_Swin"):
            raise ValueError("Only original manuscript AFE-Swin is allowed in this visualization")
        dataset, paths = test_dataset(payload, data_root)
        if hasattr(dataset, "idx"):
            labels = [dataset.ds.samples[index][1] for index in dataset.idx]
        else:
            labels = dataset.targets
        patient_map = resolve(supplied["patient_map"]) if supplied.get("patient_map") else None
        patient_mapping(paths, task[0], patient_map)
        required_classes = ["LC", "F"] if key == "breakhis_200x" else ["5_DCIS", "4_ADH"]
        if [case["class_name"] for case in supplied["cases"]] != required_classes:
            raise ValueError(f"Fig.7 case order for {key} must be {required_classes}")
        selected = select_cases(paths, labels, payload["class_names"], supplied["cases"])
        history = resolve(supplied["history_csv"]) if supplied.get("history_csv") else checkpoint.parent / "history.csv"
        if not history.is_file():
            raise ValueError(f"Real Raw/EMA training history is missing: {history}")
        validate_training_history(history, payload)
        jobs[key] = dict(checkpoint=checkpoint, data_root=data_root, payload=payload, paths=paths, labels=labels,
                         patient_map=patient_map, cases=selected, selections=supplied["cases"], history=history)
    source = config.get("parameter_source", "breakhis_200x")
    if source not in jobs:
        raise ValueError("parameter_source must select one of the two checkpoint tasks")
    repeats = config.get("bootstrap_repeats", 10000)
    if not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("bootstrap_repeats must be a positive integer")
    output = resolve(config["output_dir"]) / f"paper_figures_{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    summaries, cases, artifacts = [], [], {}
    for key, job in jobs.items():
        analysis_dir = output / key
        command = ["--checkpoint", str(job["checkpoint"]), "--data-root", str(job["data_root"]),
                   "--output", str(analysis_dir), "--device", args.device, "--bootstrap", str(repeats),
                   "--no-plots", "--examples", *job["cases"]]
        if job["patient_map"] is not None:
            command += ["--patient-map", str(job["patient_map"])]
        analyze_main(command)
        summary = json.loads((analysis_dir / "summary.json").read_text(encoding="utf-8"))
        if summary["scope"] != "complete_recorded_test_set" or summary["diagnostic_only"]:
            raise ValueError("Only complete, non-diagnostic paired analyses can enter Fig.8")
        summary["dataset"] = "BreaKHis 200×八分类" if key == "breakhis_200x" else "BRACS七分类"
        summaries.append(summary)
        manifest = json.loads((analysis_dir / "case_manifest.json").read_text(encoding="utf-8"))
        by_name = {row["image"]: row for row in manifest["cases"]}
        artifacts[key] = dict(checkpoint_sha256=summary["checkpoint_sha256"], epoch=summary["selected_epoch"],
                              cases=job["cases"], actual_pairs=[{"n": row["n"], "g": row["g"]}
                                                               for row in summary["comparisons"]])
        for name, selection in zip(job["cases"], job["selections"]):
            if name not in by_name:
                raise ValueError(f"Selected case {name} failed the prespecified geometric inclusion rule")
            case = load_measured_case(analysis_dir / by_name[name]["maps"], selection["display_label"])
            case["dataset_key"] = key
            cases.append(case)
        if args.gradcam:
            classes = ["F", "PT", "LC"] if key == "breakhis_200x" else ["2_UDH", "4_ADH", "5_DCIS"]
            gradcam_cases = select_cases(job["paths"], job["labels"], job["payload"]["class_names"],
                                         [{"class_name": name} for name in classes])
            gradcam_main(["--checkpoint", str(job["checkpoint"]), "--data-root", str(job["data_root"]),
                          "--output", str(output / f"{key}_afe_gradcam"), "--device", args.device,
                          "--target", "truth", "--images", *gradcam_cases])
    cases.sort(key=lambda case: 0 if case["dataset_key"] == "bracs_seven" else 1)
    with np.load(output / source / "measured_filter.npz", allow_pickle=False) as measured:
        plot_filter(measured["weight"], output / "Fig7_a_filter_deviation")
    plot_gate(jobs[source]["history"], jobs[source]["payload"]["epoch"], output / "Fig7_b_gate")
    plot_cases(cases, output / "Fig7_c_four_cases")
    plot_forest(summaries, output / "Fig8_paired_delta")
    write_supplementary_table(summaries, output / "Table_S1_full_paired_statistics.csv")
    write_json(output / "figures_provenance.json", {"config": config, "artifacts": artifacts,
               "parameter_source": source, "parameter_definition": "channel mean abs(W_c-1)",
               "case_response": "channel RMS of irFFT(M_b*FFT(F)*(W-1)); sqrt display only",
               "bands": "DC excluded; r=(fx^2+fy^2)^0.5/(0.5^2+0.5^2)^0.5; (0,1/3],(1/3,2/3],(2/3,1]",
               "structure_references": "gray Sobel edge and HED hematoxylin density; not clinical segmentation",
               "old_manuscript_results_reproduced": False}, exclusive=True)
    print(f"Saved measured paper panels and supplementary table to {output}")
    return output


if __name__ == "__main__":
    main()
