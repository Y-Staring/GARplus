from __future__ import annotations

import runpy
import sys
from pathlib import Path


BASE = Path("/home/yyyy/codework/GARplus")
MINER = BASE / "enumeration-discovery/GARplusMiner"
SOURCE = MINER / "apply_gar_rules_to_noisy_graph.py"


def make_script(dataset: str, noisy_csv: Path, output_dir: Path, min_conf: float, min_lift: float) -> str:
    text = SOURCE.read_text(encoding="utf-8")
    replacements = {
        'DATASET_NAME = "DDA"': f'DATASET_NAME = "{dataset}"',
        'INPUT_NOISY_CSV = DATA_DIR / "drug_disease_signed.csv"': f'INPUT_NOISY_CSV = Path(r"{noisy_csv}")',
        'OUTPUT_DIR = PROCESSED_DIR / DATASET_NAME.lower() / "gar_denoised"': f'OUTPUT_DIR = Path(r"{output_dir}")',
        'LABEL_COLUMN = "interaction_label"': 'LABEL_COLUMN = "noise_label"',
        "CHECK_LABELS: set[str] | None = None": 'CHECK_LABELS: set[str] | None = {"1"}',
        "FOCUS_EXISTING_NEGATIVE_LABELS = True": "FOCUS_EXISTING_NEGATIVE_LABELS = False",
        "MIN_RULE_CONFIDENCE = 0.8": f"MIN_RULE_CONFIDENCE = {min_conf}",
        "MIN_RULE_LIFT = 2.0": f"MIN_RULE_LIFT = {min_lift}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def main() -> None:
    out_root = Path(sys.argv[1])
    selected = {arg.upper() for arg in sys.argv[2:]}
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for dataset, test_dir, file_prefix, min_conf, min_lift in [
        ("DDA", BASE / "experiments/exp1_accuracy/DDA_test/data/noise", "drug_disease", 0.8, 2.0),
        ("TI", BASE / "experiments/exp1_accuracy/TI_test/data/noise", "gene_disease", 0.0, 0.0),
    ]:
        if selected and dataset not in selected:
            continue
        for pct in ("5pct", "10pct", "20pct"):
            jobs.append((dataset, pct, test_dir / f"{file_prefix}_{pct}.csv", min_conf, min_lift))

    for dataset, pct, noisy_csv, min_conf, min_lift in jobs:
        print(
            f"\n[NoiseOnlyJob] dataset={dataset} pct={pct} min_conf={min_conf} min_lift={min_lift} "
            f"input={noisy_csv} exists={noisy_csv.exists()}",
            flush=True,
        )
        if not noisy_csv.exists():
            continue
        tmp = out_root / f"_tmp_noise_only_{dataset.lower()}_{pct}.py"
        output_dir = out_root / dataset.lower() / pct
        tmp.write_text(make_script(dataset, noisy_csv, output_dir, min_conf, min_lift), encoding="utf-8")
        try:
            runpy.run_path(str(tmp), run_name="__main__")
        except Exception as exc:
            print(f"[NoiseOnlyError] dataset={dataset} pct={pct} error={type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
