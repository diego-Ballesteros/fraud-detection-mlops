"""Check the chosen configuration's stability, then fit, serialize and register the artifact.

Run from the repository root::

    uv run python scripts/register_model.py

Steps, in order (ADR 0010):

1. **Seed check.** The chosen LightGBM sits near the region where its family diverges
   (ERR-014), so before anything is serialized it is refitted with three other seeds on the
   five temporal folds. If any seed's mean decision metric moves more than trial 5's spread
   across folds (``TRIAL_STD``) away from trial 5's mean, the script stops: nothing is
   serialized. It also says whether the seed changed anything at all; if every metric of
   every fold is identical across seeds, the check could not have failed and proves nothing.
2. **Fit** the Pipeline on the whole reference period, every row with ``step <= 400``.
3. **Serialize** it to ``registry.ARTIFACT_PATH``.
4. **Register** it in the MLflow Model Registry, with its signature, hyperparameters and
   the data identity tags.
5. **Verify** that the version loaded back from the registry predicts exactly what the
   local artifact predicts.

Only the reference period is read.
"""

import hashlib
import sys
import time
from functools import partial
from typing import Final

import mlflow
import numpy as np

from fraud_detection.config import settings
from fraud_detection.models.estimators import ModelSpec, full_preprocessor
from fraud_detection.models.evaluation import (
    TEMPORAL,
    derived_seeds,
    features_and_labels,
    load_evaluation_frame,
)
from fraud_detection.models.registry import (
    ARTIFACT_PATH,
    CHOSEN_PARAMS,
    CHOSEN_STUDY,
    CHOSEN_TRIAL,
    MODEL_INPUT_COLUMNS,
    REGISTERED_MODEL_NAME,
    build_chosen_pipeline,
    chosen_model,
    load_pipeline,
    register_pipeline,
    save_pipeline,
)
from fraud_detection.models.search_space import OBJECTIVE_METRIC
from fraud_detection.models.tracking import base_tags, evaluate_and_log, reference_identity
from fraud_detection.tracking.mlflow_setup import configure_with_local_fallback

TRIAL_MEAN: Final[float] = 0.9969675943013213
"""Trial 5's mean decision metric over the five temporal folds, as the study stored it."""

TRIAL_STD: Final[float] = 0.0036
"""Trial 5's spread across folds (sample standard deviation), the tolerance of the check."""

CHECK_SEEDS: Final[tuple[int, ...]] = derived_seeds(4)[1:]
"""Three seeds other than ``settings.RANDOM_STATE``, the one trial 5 used."""

COMPARED: Final[tuple[str, ...]] = (OBJECTIVE_METRIC, "pr_auc", "brier", "roc_auc")


def _seed_check(shared: dict[str, str]) -> bool:
    print(f"=== 1. SEED CHECK: trial {CHOSEN_TRIAL} mean {TRIAL_MEAN:.4f}, tolerance ±{TRIAL_STD}")
    per_seed = {}
    for seed in CHECK_SEEDS:
        spec = ModelSpec("lightgbm", full_preprocessor, partial(chosen_model, seed), True)
        run, run_id = evaluate_and_log(
            spec,
            TEMPORAL,
            shared_tags=shared,
            extra_tags={"purpose": "seed_check", "check_seed": str(seed)},
            run_suffix=f"__seed{seed}",
        )
        if not run.complete:
            print(f"seed {seed}: run incomplete ({run.error or run.aborted_fold}); stopping")
            return False
        values = [fold.metrics[OBJECTIVE_METRIC] for fold in run.folds]
        per_seed[seed] = run.folds
        mean, std = float(np.mean(values)), float(np.std(values, ddof=1))
        print(
            f"seed {seed}: {OBJECTIVE_METRIC} {mean:.4f} ± {std:.4f} "
            f"folds {[round(v, 4) for v in values]} | run {run_id[:8]}"
        )
    means = {
        seed: float(np.mean([f.metrics[OBJECTIVE_METRIC] for f in folds]))
        for seed, folds in per_seed.items()
    }
    worst = max(abs(mean - TRIAL_MEAN) for mean in means.values())
    first = next(iter(per_seed.values()))
    inert = all(
        [f.metrics[name] for f in folds] == [f.metrics[name] for f in first]
        for folds in per_seed.values()
        for name in COMPARED
    )
    print(f"largest distance from trial {CHOSEN_TRIAL}: {worst:.4f} (tolerance {TRIAL_STD})")
    print(
        f"every compared metric identical across seeds: {inert}"
        + (" -> the seed changed nothing; the check could not fail" if inert else "")
    )
    return worst <= TRIAL_STD


def main() -> int:
    """Run the five steps; stop before serializing if the seed check fails."""
    # The project's standard defense (ERR-012): redirected, stdout on Windows is cp1252.
    sys.stdout.reconfigure(encoding="utf-8")
    target, failure = configure_with_local_fallback()
    if failure is not None:
        print("Remote tracking failed; using the local store. Full error:\n" + failure)
    print(f"tracking URI : {target.tracking_uri} | experiment {target.experiment_name}")
    identity = reference_identity()
    shared = base_tags(identity)

    if not _seed_check(shared):
        print("\nSEED CHECK FAILED: nothing serialized, nothing registered.")
        return 1

    print("\n=== 2. FIT ON THE WHOLE REFERENCE PERIOD")
    features, labels = features_and_labels(load_evaluation_frame())
    pipeline = build_chosen_pipeline()
    started = time.perf_counter()
    pipeline.fit(features, labels)
    fit_seconds = time.perf_counter() - started
    print(f"fitted on {len(features)} rows, {int(labels.sum())} positives, in {fit_seconds:.1f} s")

    print("\n=== 3. SERIALIZE")
    size = save_pipeline(pipeline, ARTIFACT_PATH)
    digest = hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest()
    print(
        f"{ARTIFACT_PATH.relative_to(settings.ROOT_DIR)}: {size} bytes ({size / 1024:.1f} KB), "
        f"sha256 {digest}"
    )

    print("\n=== 4. REGISTER")
    example = features[list(MODEL_INPUT_COLUMNS)].head(5).astype({"type": object})
    tags = {
        **shared,
        "chosen_study": CHOSEN_STUDY,
        "chosen_trial": str(CHOSEN_TRIAL),
        "decision": "ADR 0010",
        "artifact_sha256": digest,
        "train_rows": str(len(features)),
        "train_positives": str(int(labels.sum())),
    }
    params = {**CHOSEN_PARAMS, "random_state": settings.RANDOM_STATE, "fit_seconds": fit_seconds}
    run_id, version = register_pipeline(pipeline, example, target=target, tags=tags, params=params)
    print(f"registered {REGISTERED_MODEL_NAME} version {version} from run {run_id}")

    print("\n=== 5. VERIFY")
    rows = features[list(MODEL_INPUT_COLUMNS)].iloc[:: len(features) // 1000].head(1000)
    local = load_pipeline(ARTIFACT_PATH).predict_proba(rows)
    registered = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{version}")
    remote = registered.predict_proba(rows)
    in_memory = pipeline.predict_proba(rows)
    print(f"local artifact equals the fitted Pipeline: {np.array_equal(local, in_memory)}")
    print(f"registry version {version} equals the local artifact: {np.array_equal(remote, local)}")
    same = np.array_equal(local, in_memory) and np.array_equal(remote, local)
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
