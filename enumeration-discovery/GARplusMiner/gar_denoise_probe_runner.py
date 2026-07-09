from __future__ import annotations

import runpy
import sys
from pathlib import Path


BASE = Path("/home/yyyy/codework/GARplus")
MINER = BASE / "enumeration-discovery/GARplusMiner"
SOURCE = MINER / "apply_gar_rules_to_noisy_graph.py"


def make_script(dataset: str, noisy_csv: Path, output_dir: Path) -> str:
    text = SOURCE.read_text(encoding="utf-8")
    replacements = {
        'DATASET_NAME = "DDA"': f'DATASET_NAME = "{dataset}"',
        'INPUT_NOISY_CSV = DATA_DIR / "drug_disease_signed.csv"': f'INPUT_NOISY_CSV = Path(r"{noisy_csv}")',
        'OUTPUT_DIR = PROCESSED_DIR / DATASET_NAME.lower() / "gar_denoised"': f'OUTPUT_DIR = Path(r"{output_dir}")',
        'LABEL_COLUMN = "interaction_label"': 'LABEL_COLUMN = "noise_label"',
        "FOCUS_EXISTING_NEGATIVE_LABELS = True": "FOCUS_EXISTING_NEGATIVE_LABELS = False",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def main() -> None:
    out_root = Path(sys.argv[1])
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for dataset, test_dir, file_prefix in [
        ("DDA", BASE / "experiments/exp1_accuracy/DDA_test/data/noise", "drug_disease"),
        ("TI", BASE / "experiments/exp1_accuracy/TI_test/data/noise", "gene_disease"),
    ]:
        for pct in ("5pct", "10pct", "20pct"):
            jobs.append((dataset, pct, test_dir / f"{file_prefix}_{pct}.csv"))

    for dataset, pct, noisy_csv in jobs:
        print(f"\n[ProbeJob] dataset={dataset} pct={pct} input={noisy_csv} exists={noisy_csv.exists()}", flush=True)
        if not noisy_csv.exists():
            print(f"[ProbeSkip] missing_input dataset={dataset} pct={pct} input={noisy_csv}", flush=True)
            continue
        tmp = out_root / f"_tmp_apply_{dataset.lower()}_{pct}.py"
        output_dir = out_root / dataset.lower() / pct
        tmp.write_text(make_script(dataset, noisy_csv, output_dir), encoding="utf-8")
        try:
            runpy.run_path(str(tmp), run_name="__main__")
        except Exception as exc:
            print(f"[ProbeError] dataset={dataset} pct={pct} error={type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
