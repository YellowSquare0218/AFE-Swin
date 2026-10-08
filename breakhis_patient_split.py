from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re


SPLITS = ("train", "val", "test")
RATIOS = (0.6, 0.2, 0.2)
PROTOCOL = "breakhis_patient_stratified_60_20_20_v1"
FILENAME = re.compile(
    r"(SOB_(?:B_(?:A|F|PT|TA)|M_(?:DC|LC|MC|PC))-\d{2}-\d+[A-Z]*)"
    r"-(?P<magnification>40|100|200|400)-\d+\.png"
)


def patient_id_from_path(path):

    filename = Path(path).name
    match = FILENAME.fullmatch(filename)
    if match is None:
        raise ValueError(f"Unrecognized BreaKHis filename: {filename}")
    return match.group(1)


def build_patient_split(patient_ids, seed):
    strata = defaultdict(list)
    for patient in sorted(set(patient_ids)):
        if patient_id_from_path(patient + "-200-001.png") != patient:
            raise ValueError(f"Invalid patient identifier: {patient}")
        strata[patient.split("-")[0].split("_")[-1]].append(patient)
    if not strata or any(len(values) < 3 for values in strata.values()):
        raise ValueError("Patient-level stratification requires at least three identifiers per subtype.")
    total = sum(len(values) for values in strata.values())
    targets = [int(total * RATIOS[0]), int(total * RATIOS[1])]
    targets.append(total - sum(targets))
    if min(targets) < len(strata):
        raise ValueError("Too few patients for 60/20/20 with every subtype in all three splits.")


    ideal = {subtype: [len(values) * ratio for ratio in RATIOS] for subtype, values in strata.items()}
    allocation = {}
    for subtype, values in strata.items():
        counts = [1, 1, 1]
        for _ in range(len(values) - 3):
            destination = max(range(3), key=lambda index: ideal[subtype][index] - counts[index])
            counts[destination] += 1
        allocation[subtype] = counts
    totals = [sum(counts[index] for counts in allocation.values()) for index in range(3)]
    while totals != targets:
        moves = []
        for subtype, counts in allocation.items():
            for source in range(3):
                if totals[source] <= targets[source] or counts[source] <= 1:
                    continue
                for destination in range(3):
                    if totals[destination] >= targets[destination]:
                        continue
                    cost = (2 * (counts[destination] - ideal[subtype][destination])
                            - 2 * (counts[source] - ideal[subtype][source]) + 2)
                    moves.append((cost, subtype, source, destination))
        if not moves:
            raise ValueError("Cannot allocate patient counts without losing subtype coverage.")
        _, subtype, source, destination = min(moves)
        allocation[subtype][source] -= 1
        allocation[subtype][destination] += 1
        totals[source] -= 1
        totals[destination] += 1

    rng = random.Random(seed)
    result = {name: [] for name in SPLITS}
    for subtype, patients in strata.items():
        shuffled = patients.copy()
        rng.shuffle(shuffled)
        offset = 0
        for index, name in enumerate(SPLITS):
            count = allocation[subtype][index]
            result[name].extend(shuffled[offset:offset + count])
            offset += count
    return {name: sorted(values) for name, values in result.items()}


def split_breakhis_samples(samples, data_roots, seed, manifest_path, expected_magnification=None):

    if not samples:
        raise ValueError("No BreaKHis image samples were provided.")
    available_roots = [Path(root) for root in data_roots if Path(root).is_dir()]
    if not available_roots:
        raise FileNotFoundError("No configured BreaKHis data roots exist.")
    patient_ids = {patient_id_from_path(path) for root in available_roots
                   for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".png"}
    assignments = build_patient_split(patient_ids, seed)
    manifest = {
        "protocol": PROTOCOL,
        "seed": seed,
        "ratios": list(RATIOS),
        "rounding": "floor(0.6*N), floor(0.2*N), remainder assigned to test",
        "grouping": "Authors' complete case identifier: BIOPSY_CLASS_SUBTYPE-YEAR-SLIDE_ID; suffix retained",
        "stratification": "Filename subtype; at least one patient per subtype per split",
        "patient_counts": {name: len(values) for name, values in assignments.items()},
        "patients": assignments,
    }
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("Existing patient split manifest does not match the seed or patient inventory. "
                             "Restore the original inventory or choose a new manifest path; it will not be overwritten.")
    membership = {patient: name for name, patients in assignments.items() for patient in patients}
    indices = {name: [] for name in SPLITS}
    current_patients = {name: set() for name in SPLITS}
    image_inventory = []
    for index, (path, label) in enumerate(samples):
        patient = patient_id_from_path(path)
        if expected_magnification is not None:
            actual = int(FILENAME.fullmatch(Path(path).name).group("magnification"))
            if actual != expected_magnification:
                raise ValueError(f"BreaKHis magnification mismatch: expected {expected_magnification}X, "
                                 f"found {actual}X in {path}. Correct Data_dir or the dataset; "
                                 "do not relabel or move raw images automatically.")
        if patient not in membership:
            raise ValueError(f"Sample patient is absent from the shared manifest: {patient}")
        name = membership[patient]
        indices[name].append(index)
        current_patients[name].add(patient)
        image_inventory.append((Path(path).name, int(label)))
    if any(not selected for selected in indices.values()):
        raise ValueError("The current magnification has an empty patient-level split.")
    if not manifest_path.exists():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
    metadata = dict(manifest)
    metadata.update(
        manifest_path=str(manifest_path.resolve()),
        expected_magnification=expected_magnification,
        available_data_roots=[str(root.resolve()) for root in available_roots],
        current_patient_counts={name: len(values) for name, values in current_patients.items()},
        image_counts={name: len(values) for name, values in indices.items()},
        subtype_patient_counts={name: dict(Counter(patient.split("-")[0].split("_")[-1]
                                                  for patient in values))
                                for name, values in assignments.items()},
        image_inventory_sha256=hashlib.sha256(json.dumps(sorted(image_inventory)).encode()).hexdigest(),
    )
    return indices, metadata
