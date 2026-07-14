import argparse
import random
import shutil
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = REPO_ROOT / "experiments" / "exp1_accuracy" / "NBFNet" / "data_updated"
NOISE_RATIOS = (0.05, 0.10, 0.20)
BASE_SEED = 20260713

# Stronger, staged corruption schedule.
# Prefixes are cumulative: 10% contains 5%, 20% contains 10%.
ZERO_FRACTION_BY_RATIO = {
    0.05: 0.20,
    0.10: 0.30,
    0.20: 0.40,
}


def load_triples(path):
    triples = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            head, relation, tail = line.split("\t")
            triples.append([head, int(relation), tail])
    return triples


def save_triples(path, triples):
    with path.open("w") as handle:
        for head, relation, tail in triples:
            handle.write(f"{head}\t{relation}\t{tail}\n")


def tie_break_sorted(indices, scores, rng):
    shuffled = list(indices)
    rng.shuffle(shuffled)
    return sorted(shuffled, key=lambda idx: scores[idx], reverse=True)


def first_index_per_label(triples):
    first = {}
    for idx, (_, relation, _) in enumerate(triples):
        if relation not in first:
            first[relation] = idx
    return first


def build_monotonic_flip_plan(triples, rng):
    counts = Counter(relation for _, relation, _ in triples)
    train_size = len(triples)

    target_by_ratio = {ratio: int(round(train_size * ratio)) for ratio in NOISE_RATIOS}
    max_target = max(target_by_ratio.values())

    # Protect the first occurrence of each relation to avoid relation-vocab order drift.
    protected = set(first_index_per_label(triples).values())

    node_freq = Counter()
    for head, _, tail in triples:
        node_freq[head] += 1
        node_freq[tail] += 1
    scores = [node_freq[head] + node_freq[tail] for head, _, tail in triples]

    pos_candidates = [
        idx for idx, (_, relation, _) in enumerate(triples)
        if relation in (1, 2) and idx not in protected
    ]
    zero_candidates = [
        idx for idx, (_, relation, _) in enumerate(triples)
        if relation == 0 and idx not in protected
    ]
    pos_ranked = tie_break_sorted(pos_candidates, scores, rng)
    zero_ranked = tie_break_sorted(zero_candidates, scores, rng)

    if max_target > len(pos_ranked) + len(zero_ranked):
        raise ValueError(
            f"Not enough candidates to flip: need {max_target}, have {len(pos_ranked) + len(zero_ranked)}"
        )

    # Decide how many 0-label flips are used at each noise level.
    zero_target_by_ratio = {
        ratio: min(int(round(target_by_ratio[ratio] * ZERO_FRACTION_BY_RATIO[ratio])), len(zero_ranked))
        for ratio in NOISE_RATIOS
    }
    # Enforce cumulative monotonicity on the zero quota.
    prev = 0
    for ratio in sorted(NOISE_RATIOS):
        zero_target_by_ratio[ratio] = max(prev, zero_target_by_ratio[ratio])
        prev = zero_target_by_ratio[ratio]

    ordered = []
    pos_cursor = 0
    zero_cursor = 0
    prefix_targets = [target_by_ratio[ratio] for ratio in sorted(NOISE_RATIOS)]
    prefix_zero_targets = [zero_target_by_ratio[ratio] for ratio in sorted(NOISE_RATIOS)]

    for step, total_target in enumerate(prefix_targets):
        target_zero = prefix_zero_targets[step]
        current_total = len(ordered)
        need_total = total_target - current_total
        if need_total <= 0:
            continue

        current_zero = zero_cursor
        need_zero = max(0, target_zero - current_zero)
        need_pos = need_total - need_zero

        if pos_cursor + need_pos > len(pos_ranked):
            short = (pos_cursor + need_pos) - len(pos_ranked)
            need_pos -= short
            need_zero += short
        if zero_cursor + need_zero > len(zero_ranked):
            short = (zero_cursor + need_zero) - len(zero_ranked)
            need_zero -= short
            need_pos += short
        if pos_cursor + need_pos > len(pos_ranked) or zero_cursor + need_zero > len(zero_ranked):
            raise ValueError("Unable to satisfy flip quotas for monotonic schedule")

        ordered.extend(pos_ranked[pos_cursor: pos_cursor + need_pos])
        pos_cursor += need_pos
        ordered.extend(zero_ranked[zero_cursor: zero_cursor + need_zero])
        zero_cursor += need_zero

    if len(ordered) < max_target:
        remaining_needed = max_target - len(ordered)
        remain_pos = pos_ranked[pos_cursor:]
        remain_zero = zero_ranked[zero_cursor:]
        remain_mix = remain_pos + remain_zero
        ordered.extend(remain_mix[:remaining_needed])

    return {
        "ordered_indices": ordered[:max_target],
        "target_by_ratio": target_by_ratio,
        "counts": dict(sorted(counts.items())),
    }


def apply_flip(relation, minority_nonzero_label):
    if relation in (1, 2):
        return 0
    return minority_nonzero_label


def build_noisy_split(source_dir, target_dir, ratio, ordered_indices, target_by_ratio):
    train_path = source_dir / "train.txt"
    valid_path = source_dir / "valid.txt"
    test_path = source_dir / "test.txt"

    triples = load_triples(train_path)
    target_flips = target_by_ratio[ratio]
    flip_indices = ordered_indices[:target_flips]

    count_1 = sum(1 for _, relation, _ in triples if relation == 1)
    count_2 = sum(1 for _, relation, _ in triples if relation == 2)
    minority_nonzero_label = 1 if count_1 < count_2 else 2

    noisy_triples = [triple[:] for triple in triples]
    transition_counter = Counter()
    for index in flip_indices:
        old_relation = noisy_triples[index][1]
        new_relation = apply_flip(old_relation, minority_nonzero_label)
        noisy_triples[index][1] = new_relation
        transition_counter[(old_relation, new_relation)] += 1

    target_dir.mkdir(parents=True, exist_ok=True)
    save_triples(target_dir / "train.txt", noisy_triples)
    shutil.copy2(valid_path, target_dir / "valid.txt")
    shutil.copy2(test_path, target_dir / "test.txt")

    before_counts = Counter(relation for _, relation, _ in triples)
    after_counts = Counter(relation for _, relation, _ in noisy_triples)
    return {
        "train_size": len(triples),
        "target_flips": target_flips,
        "actual_flips": len(flip_indices),
        "transitions": {
            f"{old}->{new}": count for (old, new), count in sorted(transition_counter.items())
        },
        "before": dict(sorted(before_counts.items())),
        "after": dict(sorted(after_counts.items())),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate cumulative 5/10/20 percent label-flip noise for TI and DDA."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory containing my_dataset_ti and my_dataset_dda.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    dataset_names = ("my_dataset_ti", "my_dataset_dda")
    for dataset_index, dataset_name in enumerate(dataset_names):
        source_dir = data_root / dataset_name
        if not (source_dir / "train.txt").is_file():
            raise FileNotFoundError(
                f"Missing dataset split: {source_dir / 'train.txt'}. "
                "Pass the NBFNet data_updated directory with --data-root."
            )
        base_triples = load_triples(source_dir / "train.txt")
        flip_plan = build_monotonic_flip_plan(
            triples=base_triples,
            rng=random.Random(BASE_SEED + dataset_index * 1000),
        )

        for ratio in NOISE_RATIOS:
            ratio_pct = int(ratio * 100)
            target_dir = data_root / f"{dataset_name}_noise_{ratio_pct}pct"
            summary = build_noisy_split(
                source_dir=source_dir,
                target_dir=target_dir,
                ratio=ratio,
                ordered_indices=flip_plan["ordered_indices"],
                target_by_ratio=flip_plan["target_by_ratio"],
            )
            print(
                f"{dataset_name} {ratio_pct:02d}% -> {target_dir}\n"
                f"  flips: {summary['actual_flips']}/{summary['train_size']}\n"
                f"  transitions: {summary['transitions']}\n"
                f"  before: {summary['before']}\n"
                f"  after:  {summary['after']}\n"
            )


if __name__ == "__main__":
    main()
