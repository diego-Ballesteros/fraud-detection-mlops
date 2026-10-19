"""What the head of a permuted model's ranking holds: the measurement behind ERR-013.

Run from the repository root::

    uv run python scripts/diagnose_head.py
    uv run python scripts/diagnose_head.py --family lightgbm --fold 2 --seed 42

Refits one fold of the permutation test (temporal splitter, training labels permuted with
``--seed``) by the same code path as the harness, then counts the positives, and those with
the signature, among the top ``k`` rows. Defaults: the fold that failed the single
permutation test, LightGBM, fold 2, seed ``settings.RANDOM_STATE``. Its PR-AUC must be
0.011227402, the value logged by that test; any other value means another model.

Only the reference period is read.
"""

import argparse
import sys
from typing import Final

import numpy as np
from sklearn.metrics import average_precision_score

from fraud_detection.config import settings
from fraud_detection.features.builder import (
    DERIVATION_INPUTS,
    SIGNATURE_FEATURE,
    amount_equals_origin_balance,
)
from fraud_detection.models.estimators import MODEL_SPECS
from fraud_detection.models.evaluation import (
    SPLITTERS,
    TEMPORAL,
    load_evaluation_frame,
    ranking_head,
    score_one_fold,
)

HEAD_SIZES: Final[tuple[int, ...]] = (10, 50, 100, 1000)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", choices=list(MODEL_SPECS), default="lightgbm")
    parser.add_argument("--fold", type=int, default=2)
    parser.add_argument("--seed", type=int, default=settings.RANDOM_STATE)
    return parser.parse_args()


def main() -> int:
    """Refit the fold and report the head of its ranking."""
    # The project's standard defense (ERR-012): redirected, stdout on Windows is cp1252.
    sys.stdout.reconfigure(encoding="utf-8")
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
    print(f"{arguments.family}, temporal fold {arguments.fold}, permutation seed {arguments.seed}")
    print(f"PR-AUC {average_precision_score(y, scores):.9f}")
    print(
        f"validation rows {len(y)}, positives {int(y.sum())}, "
        f"positives with signature {int(y[signature].sum())}"
    )
    for head in ranking_head(y, scores, signature, HEAD_SIZES):
        print(
            f"top {head.k:5d}: flagged {head.flagged:6d}  positives {head.positives:3d}  "
            f"with signature {head.marked_positives:3d}  "
            f"precision {head.positives / head.flagged:.4f}"
        )
    ranks = np.flatnonzero(y[np.argsort(-scores, kind="stable")]) + 1
    percentile = ranks / len(y)
    print(
        f"rank of the positives, as a share of the ranking: median {np.median(percentile):.3f}; "
        f"share of positives in the bottom half {np.mean(percentile > 0.5):.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
