"""Where a permuted model's PR-AUC comes from, block by block: the mechanism of ERR-013.

Run from the repository root::

    uv run python scripts/diagnose_ap.py
    uv run python scripts/diagnose_ap.py --family lightgbm --fold 2 --seed 42

Refits one fold of the permutation test by the same code path as the harness and splits
its average precision into the contribution of each block of tied scores. Defaults as in
``diagnose_head.py``: LightGBM, temporal fold 2, permutation seed
``settings.RANDOM_STATE``, whose PR-AUC must be 0.011227402. For a tree model it also
counts the splits on the signature feature.

Only the reference period is read.
"""

import argparse
import sys
from typing import Final

import pandas as pd
from sklearn.metrics import average_precision_score

from fraud_detection.config import settings
from fraud_detection.features.builder import (
    DERIVATION_INPUTS,
    SIGNATURE_FEATURE,
    amount_equals_origin_balance,
)
from fraud_detection.models.estimators import MODEL_SPECS, splits_on
from fraud_detection.models.evaluation import (
    SPLITTERS,
    TEMPORAL,
    average_precision_by_block,
    load_evaluation_frame,
    score_one_fold,
)

TOP_BLOCKS: Final[int] = 10
RANKING_BANDS: Final[dict[str, tuple[float, float]]] = {
    "top 0.1%": (0.0, 0.001),
    "0.1-1%": (0.001, 0.01),
    "1-10%": (0.01, 0.10),
    "10-50%": (0.10, 0.50),
    "50-90%": (0.50, 0.90),
    "bottom 10%": (0.90, 1.0),
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", choices=list(MODEL_SPECS), default="lightgbm")
    parser.add_argument("--fold", type=int, default=2)
    parser.add_argument("--seed", type=int, default=settings.RANDOM_STATE)
    return parser.parse_args()


def main() -> int:
    """Refit the fold and decompose its average precision."""
    # The project's standard defense (ERR-012): redirected, stdout on Windows is cp1252.
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 200)
    arguments = _arguments()
    spec = MODEL_SPECS[arguments.family]
    fold = score_one_fold(
        load_evaluation_frame(),
        SPLITTERS[TEMPORAL](),
        arguments.fold,
        spec.preprocessor,
        spec.model,
        shuffle_training_labels=True,
        shuffle_seed=arguments.seed,
    )
    y, scores = fold.labels, fold.scores
    inputs = list(DERIVATION_INPUTS[SIGNATURE_FEATURE])
    signature = amount_equals_origin_balance(fold.validation[inputs]).to_numpy()
    blocks = average_precision_by_block(y, scores)
    total = float(blocks["contribution"].sum())
    print(f"{arguments.family}, temporal fold {arguments.fold}, permutation seed {arguments.seed}")
    print(
        f"PR-AUC {average_precision_score(y, scores):.9f}; sum of block contributions {total:.9f}"
    )
    print(
        f"score blocks {len(blocks)}; blocks holding a positive {(blocks['positives'] > 0).sum()}"
    )

    signature_positives = [
        int((y.astype(bool) & signature & (scores == value)).sum()) for value in blocks["threshold"]
    ]
    blocks["signature_positives"] = signature_positives
    blocks["share"] = blocks["contribution"] / total
    top = blocks.sort_values("share", ascending=False).head(TOP_BLOCKS).copy()
    top["cumulative_share"] = top["share"].cumsum()
    columns = [
        "share",
        "cumulative_share",
        "threshold",
        "rows",
        "positives",
        "signature_positives",
        "flagged",
        "precision",
        "recall",
    ]
    print(f"\nThe {TOP_BLOCKS} blocks that contribute most to average precision:")
    print(top[columns].to_string(index=False, float_format=lambda value: f"{value:.6g}"))

    largest = scores == top["threshold"].iloc[0]
    members = fold.validation[largest]
    print(
        f"\nThe largest contributor: {int(largest.sum())} rows, {int(y[largest].sum())} positives, "
        f"{int((y.astype(bool) & largest & signature).sum())} positives with signature"
    )
    print("  types:", {k: v for k, v in members["type"].value_counts().to_dict().items() if v})
    for column in ("amount", "oldbalanceOrg", "oldbalanceDest"):
        print(
            f"  {column}: min {members[column].min():.2f} median {members[column].median():.2f} "
            f"max {members[column].max():.2f}"
        )

    print("\nAverage precision by band of the ranking (band of the block's last row):")
    end_share = blocks["flagged"] / len(y)
    for band, (low, high) in RANKING_BANDS.items():
        rows = blocks[(end_share > low) & (end_share <= high)]
        print(
            f"  {band:10}: positives {int(rows['positives'].sum()):3d}  "
            f"share of AP {rows['contribution'].sum() / total:.3f}"
        )

    splits = splits_on(fold.estimator, SIGNATURE_FEATURE)
    if splits is None:
        print(f"\n{arguments.family} has no trees: no split count")
    else:
        print(f"\nsplits on {SIGNATURE_FEATURE} over every tree: {splits}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
