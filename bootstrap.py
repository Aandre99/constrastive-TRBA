#!/usr/bin/env python3
"""
Bootstrap statistical analysis for the ALPR 2x2 factorial experiment.

Input:
    evaluation_results/bootstrap_predictions.csv

Expected columns:
    run_id, config, seed, img, label, pred, correct, edit_dist,
    gt_length, norm_ed, confidence, vehicle_type

The script:
1. Validates that all runs contain the same test plates.
2. Computes point estimates for Accuracy and CER.
3. Performs non-parametric bootstrap resampling at plate level.
4. Uses paired bootstrap for comparisons between models.
5. Computes the 2x2 factorial effects:
       Contrastive effect without 2D attention
       Contrastive effect with 2D attention
       Attention effect without contrastive supervision
       Attention effect with contrastive supervision
       Interaction effect
6. Repeats the bootstrap independently for every seed.
7. Saves:
       bootstrap_per_seed.csv
       bootstrap_summary.csv
       bootstrap_effects_per_seed.csv
       bootstrap_effects_summary.csv
       bootstrap_results.json

The bootstrap CIs describe test-set sampling uncertainty.
Variation between the five seeds is kept separate and summarized
using mean +/- standard deviation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CONFIGS = ("TRBA", "CTRBA", "TRB2A", "CTRB2A")

EXPECTED_CONFIGS = {
    "TRBA",
    "CTRBA",
    "TRB2A",
    "CTRB2A",
}


def percentile_ci(values, alpha=0.05):
    """Percentile bootstrap confidence interval."""
    return (
        float(np.percentile(values, 100 * alpha / 2)),
        float(np.percentile(values, 100 * (1 - alpha / 2))),
    )


def accuracy(correct):
    return float(np.mean(correct))


def cer(edit_dist, gt_length):
    denominator = np.sum(gt_length)
    if denominator <= 0:
        return float("nan")
    return float(np.sum(edit_dist) / denominator)


def validate_input(df):
    required = {
        "run_id",
        "config",
        "seed",
        "img",
        "label",
        "pred",
        "correct",
        "edit_dist",
        "gt_length",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing))
        )

    df = df.copy()

    df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
    df["correct"] = pd.to_numeric(df["correct"], errors="raise").astype(int)
    df["edit_dist"] = pd.to_numeric(df["edit_dist"], errors="raise")
    df["gt_length"] = pd.to_numeric(df["gt_length"], errors="raise")

    if not df["correct"].isin([0, 1]).all():
        raise ValueError("Column 'correct' must contain only 0/1 values.")

    if (df["edit_dist"] < 0).any():
        raise ValueError("'edit_dist' cannot contain negative values.")

    if (df["gt_length"] <= 0).any():
        raise ValueError("'gt_length' must be > 0.")

    unknown_configs = set(df["config"].dropna().unique()) - EXPECTED_CONFIGS
    if unknown_configs:
        raise ValueError(
            "Unknown configurations found: "
            + ", ".join(sorted(map(str, unknown_configs)))
        )

    duplicated = df.duplicated(subset=["run_id", "img"])
    if duplicated.any():
        examples = (
            df.loc[duplicated, ["run_id", "img"]]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "Duplicate (run_id, img) rows found. Examples: "
            + repr(examples)
        )

    return df


def validate_test_set_alignment(df):
    """
    Verify that every run has the same test images.

    This is essential for paired bootstrap comparisons.
    """
    run_to_imgs = {
        run_id: set(group["img"].astype(str))
        for run_id, group in df.groupby("run_id", sort=False)
    }

    if not run_to_imgs:
        raise ValueError("No runs found.")

    reference_run, reference_imgs = next(iter(run_to_imgs.items()))

    for run_id, imgs in run_to_imgs.items():
        if imgs != reference_imgs:
            missing = reference_imgs - imgs
            extra = imgs - reference_imgs

            raise ValueError(
                f"Test-set mismatch for run {run_id!r}.\n"
                f"Reference run: {reference_run!r}\n"
                f"Missing images: {len(missing)}\n"
                f"Extra images: {len(extra)}"
            )

    print(
        f"[OK] Test-set alignment verified: "
        f"{len(reference_imgs)} plates per run."
    )


def prepare_seed_arrays(seed_df):
    """
    Return arrays indexed by the same plate order for the four configurations.
    """
    runs = (
        seed_df[["run_id", "config"]]
        .drop_duplicates()
        .sort_values(["config", "run_id"])
    )

    arrays = {}

    for config in CONFIGS:
        rows = runs[runs["config"] == config]

        if len(rows) != 1:
            raise ValueError(
                f"Expected exactly one run for config {config!r} "
                f"for a given seed, found {len(rows)}."
            )

        run_id = rows.iloc[0]["run_id"]

        run_df = (
            seed_df[seed_df["run_id"] == run_id]
            .sort_values("img")
            .reset_index(drop=True)
        )

        arrays[config] = {
            "run_id": np.array([run_id], dtype=object),
            "img": run_df["img"].astype(str).to_numpy(),
            "correct": run_df["correct"].to_numpy(dtype=np.float64),
            "edit_dist": run_df["edit_dist"].to_numpy(dtype=np.float64),
            "gt_length": run_df["gt_length"].to_numpy(dtype=np.float64),
        }

    reference_imgs = arrays["TRBA"]["img"]

    for config in CONFIGS:
        if not np.array_equal(arrays[config]["img"], reference_imgs):
            raise ValueError(
                f"Plate ordering mismatch inside seed for {config}."
            )

    return arrays


def bootstrap_single_seed(arrays, n_bootstrap, batch_size, rng, confidence_level):
    """
    Perform bootstrap for one seed.

    Bootstrap samples are generated once per replicate and reused for all
    four configurations. This makes all model comparisons paired.
    """
    n = len(arrays["TRBA"]["correct"])

    alpha = 1.0 - confidence_level

    metric_samples = {
        config: {
            "accuracy": np.empty(n_bootstrap, dtype=np.float64),
            "cer": np.empty(n_bootstrap, dtype=np.float64),
        }
        for config in CONFIGS
    }

    effect_names = [
        "contrastive_effect_1d",
        "contrastive_effect_2d",
        "attention_effect_no_contrastive",
        "attention_effect_contrastive",
        "interaction",
    ]

    effect_samples = {
        effect: {
            "accuracy": np.empty(n_bootstrap, dtype=np.float64),
            "cer": np.empty(n_bootstrap, dtype=np.float64),
        }
        for effect in effect_names
    }

    processed = 0

    while processed < n_bootstrap:
        current = min(batch_size, n_bootstrap - processed)

        # Shape: (current, n_plates)
        indices = rng.integers(
            low=0,
            high=n,
            size=(current, n),
            dtype=np.int32,
        )

        acc = {}
        cer_values = {}

        for config in CONFIGS:
            c = arrays[config]["correct"]
            ed = arrays[config]["edit_dist"]
            gl = arrays[config]["gt_length"]

            acc_values = np.mean(c[indices], axis=1)

            edit_sums = np.sum(ed[indices], axis=1)
            gt_sums = np.sum(gl[indices], axis=1)
            cer_values_config = edit_sums / gt_sums

            acc[config] = acc_values
            cer_values[config] = cer_values_config

            metric_samples[config]["accuracy"][
                processed : processed + current
            ] = acc_values

            metric_samples[config]["cer"][
                processed : processed + current
            ] = cer_values_config

        # Paired factorial effects
        #
        # Contrastive effect in 1D:
        #     CTRBA - TRBA
        #
        # Contrastive effect in 2D:
        #     CTRB2A - TRB2A
        #
        # Attention effect without contrastive:
        #     TRB2A - TRBA
        #
        # Attention effect with contrastive:
        #     CTRB2A - CTRBA
        #
        # Interaction:
        #     (CTRB2A - TRB2A) - (CTRBA - TRBA)
        #
        # Accuracy is additive, so effects can be computed directly.
        effect_samples["contrastive_effect_1d"]["accuracy"][
            processed : processed + current
        ] = acc["CTRBA"] - acc["TRBA"]

        effect_samples["contrastive_effect_2d"]["accuracy"][
            processed : processed + current
        ] = acc["CTRB2A"] - acc["TRB2A"]

        effect_samples["attention_effect_no_contrastive"]["accuracy"][
            processed : processed + current
        ] = acc["TRB2A"] - acc["TRBA"]

        effect_samples["attention_effect_contrastive"]["accuracy"][
            processed : processed + current
        ] = acc["CTRB2A"] - acc["CTRBA"]

        effect_samples["interaction"]["accuracy"][
            processed : processed + current
        ] = (
            (acc["CTRB2A"] - acc["TRB2A"])
            - (acc["CTRBA"] - acc["TRBA"])
        )

        # CER is a ratio, therefore we must calculate each CER from the
        # resampled numerator/denominator rather than subtracting CIs or
        # averaging per-plate CER values.
        cer_effects = {
            "contrastive_effect_1d": (
                cer_values["CTRBA"] - cer_values["TRBA"]
            ),
            "contrastive_effect_2d": (
                cer_values["CTRB2A"] - cer_values["TRB2A"]
            ),
            "attention_effect_no_contrastive": (
                cer_values["TRB2A"] - cer_values["TRBA"]
            ),
            "attention_effect_contrastive": (
                cer_values["CTRB2A"] - cer_values["CTRBA"]
            ),
            "interaction": (
                (cer_values["CTRB2A"] - cer_values["TRB2A"])
                - (cer_values["CTRBA"] - cer_values["TRBA"])
            ),
        }

        for effect, values in cer_effects.items():
            effect_samples[effect]["cer"][
                processed : processed + current
            ] = values

        processed += current

        if processed % max(batch_size * 5, 1) == 0 or processed == n_bootstrap:
            print(
                f"    bootstrap: {processed:,}/{n_bootstrap:,}",
                end="\r",
                flush=True,
            )

    print("")

    rows = []

    for config in CONFIGS:
        for metric in ("accuracy", "cer"):
            samples = metric_samples[config][metric]
            ci_low, ci_high = percentile_ci(samples, alpha)

            rows.append(
                {
                    "config": config,
                    "metric": metric,
                    "point_estimate": float(
                        accuracy(arrays[config]["correct"])
                        if metric == "accuracy"
                        else cer(
                            arrays[config]["edit_dist"],
                            arrays[config]["gt_length"],
                        )
                    ),
                    "bootstrap_mean": float(np.mean(samples)),
                    "bootstrap_sd": float(np.std(samples, ddof=1)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "confidence_level": confidence_level,
                    "n_bootstrap": n_bootstrap,
                    "n_plates": n,
                }
            )

    metric_df = pd.DataFrame(rows)

    effect_rows = []

    for effect in effect_names:
        for metric in ("accuracy", "cer"):
            samples = effect_samples[effect][metric]
            ci_low, ci_high = percentile_ci(samples, alpha)

            effect_rows.append(
                {
                    "effect": effect,
                    "metric": metric,
                    "bootstrap_mean": float(np.mean(samples)),
                    "bootstrap_sd": float(np.std(samples, ddof=1)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "confidence_level": confidence_level,
                    "n_bootstrap": n_bootstrap,
                    "n_plates": n,
                }
            )

    effects_df = pd.DataFrame(effect_rows)

    return metric_df, effects_df


def compute_point_estimates(df):
    rows = []

    grouped = df.groupby(["seed", "run_id", "config"], sort=True)

    for (seed, run_id, config), group in grouped:
        correct = group["correct"].to_numpy(dtype=float)
        edit_dist = group["edit_dist"].to_numpy(dtype=float)
        gt_length = group["gt_length"].to_numpy(dtype=float)

        rows.append(
            {
                "seed": int(seed),
                "run_id": run_id,
                "config": config,
                "n_plates": len(group),
                "accuracy": accuracy(correct),
                "cer": cer(edit_dist, gt_length),
                "total_edit_distance": float(np.sum(edit_dist)),
                "total_gt_chars": int(np.sum(gt_length)),
            }
        )

    return pd.DataFrame(rows)


def summarize_across_seeds(per_seed, value_columns):
    """
    Summarize point estimates / bootstrap intervals across the five seeds.

    Important:
    - The mean +/- SD across seeds describes training-run variability.
    - It is NOT a pooled bootstrap CI.
    """
    rows = []

    grouped = per_seed.groupby(["config", "metric"], sort=True)

    for (config, metric), group in grouped:
        row = {
            "config": config,
            "metric": metric,
            "n_seeds": len(group),
        }

        for col in value_columns:
            values = group[col].to_numpy(dtype=float)
            row[f"{col}_mean"] = float(np.mean(values))
            row[f"{col}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap analysis for the ALPR 2x2 factorial experiment."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("evaluation_results/bootstrap_predictions.csv"),
        help="Input consolidated prediction CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results/bootstrap"),
        help="Directory where bootstrap results are saved.",
    )

    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap replicates per seed.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=250,
        help="Number of bootstrap replicates processed at once.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Bootstrap confidence level.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=20260926,
        help="Random seed used by the bootstrap generator.",
    )

    args = parser.parse_args()

    if args.n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be > 0.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")

    if not 0 < args.confidence < 1:
        raise ValueError("--confidence must be between 0 and 1.")

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}\n"
            "Run evaluate_mlflow_bootstrap.py first."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("ALPR Bootstrap Analysis")
    print("=" * 72)
    print(f"Input:          {args.input}")
    print(f"Output:         {args.output_dir}")
    print(f"Bootstrap B:    {args.n_bootstrap:,}")
    print(f"Batch size:     {args.batch_size:,}")
    print(f"Confidence:     {args.confidence:.1%}")
    print(f"Random seed:    {args.random_seed}")
    print("=" * 72)

    df = pd.read_csv(args.input)

    print(f"[INFO] Loaded {len(df):,} rows.")

    df = validate_input(df)

    validate_test_set_alignment(df)

    point_df = compute_point_estimates(df)
    point_df.to_csv(
        args.output_dir / "point_estimates.csv",
        index=False,
    )

    seeds = sorted(df["seed"].unique().tolist())

    print(f"[INFO] Seeds found: {seeds}")

    if len(seeds) != 5:
        print(
            f"[WARNING] Expected 5 seeds based on the experimental design, "
            f"but found {len(seeds)}."
        )

    per_seed_metric_results = []
    per_seed_effect_results = []

    master_rng = np.random.default_rng(args.random_seed)

    for seed in seeds:
        print("")
        print("-" * 72)
        print(f"Seed {seed}")
        print("-" * 72)

        seed_df = df[df["seed"] == seed].copy()

        configs_found = set(seed_df["config"].unique())

        if configs_found != EXPECTED_CONFIGS:
            missing = EXPECTED_CONFIGS - configs_found
            extra = configs_found - EXPECTED_CONFIGS

            raise ValueError(
                f"Seed {seed} does not contain exactly the four "
                f"experimental configurations.\n"
                f"Missing: {sorted(missing)}\n"
                f"Extra: {sorted(extra)}"
            )

        arrays = prepare_seed_arrays(seed_df)

        n_plates = len(arrays["TRBA"]["correct"])

        if n_plates != 9800:
            print(
                f"[WARNING] Seed {seed}: expected 9800 plates, "
                f"found {n_plates}."
            )

        # Independent bootstrap stream for each seed.
        bootstrap_seed = int(master_rng.integers(0, 2**63 - 1))
        rng = np.random.default_rng(bootstrap_seed)

        print(
            f"[INFO] Seed {seed}: {n_plates:,} plates, "
            f"bootstrap RNG seed = {bootstrap_seed}"
        )

        metric_results, effect_results = bootstrap_single_seed(
            arrays=arrays,
            n_bootstrap=args.n_bootstrap,
            batch_size=args.batch_size,
            rng=rng,
            confidence_level=args.confidence,
        )

        metric_results.insert(0, "seed", seed)

        run_map = {
            config: str(arrays[config]["run_id"][0])
            for config in CONFIGS
        }

        metric_results["run_id"] = metric_results["config"].map(run_map)

        effect_results.insert(0, "seed", seed)

        per_seed_metric_results.append(metric_results)
        per_seed_effect_results.append(effect_results)

    per_seed_metrics = pd.concat(
        per_seed_metric_results,
        ignore_index=True,
    )

    per_seed_effects = pd.concat(
        per_seed_effect_results,
        ignore_index=True,
    )

    # Reorder columns for readability.
    metric_columns = [
        "seed",
        "run_id",
        "config",
        "metric",
        "point_estimate",
        "bootstrap_mean",
        "bootstrap_sd",
        "ci_low",
        "ci_high",
        "confidence_level",
        "n_bootstrap",
        "n_plates",
    ]

    per_seed_metrics = per_seed_metrics[metric_columns]

    effect_columns = [
        "seed",
        "effect",
        "metric",
        "bootstrap_mean",
        "bootstrap_sd",
        "ci_low",
        "ci_high",
        "confidence_level",
        "n_bootstrap",
        "n_plates",
    ]

    per_seed_effects = per_seed_effects[effect_columns]

    per_seed_metrics.to_csv(
        args.output_dir / "bootstrap_per_seed.csv",
        index=False,
    )

    per_seed_effects.to_csv(
        args.output_dir / "bootstrap_effects_per_seed.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Summary across the five training seeds
    # ------------------------------------------------------------------

    summary_rows = []

    for (config, metric), group in per_seed_metrics.groupby(
        ["config", "metric"], sort=True
    ):
        point_values = group["point_estimate"].to_numpy(float)
        ci_low_values = group["ci_low"].to_numpy(float)
        ci_high_values = group["ci_high"].to_numpy(float)

        summary_rows.append(
            {
                "config": config,
                "metric": metric,
                "n_seeds": len(group),
                "point_mean": float(np.mean(point_values)),
                "point_sd": (
                    float(np.std(point_values, ddof=1))
                    if len(point_values) > 1
                    else 0.0
                ),
                "bootstrap_ci_low_mean": float(np.mean(ci_low_values)),
                "bootstrap_ci_low_sd": (
                    float(np.std(ci_low_values, ddof=1))
                    if len(ci_low_values) > 1
                    else 0.0
                ),
                "bootstrap_ci_high_mean": float(np.mean(ci_high_values)),
                "bootstrap_ci_high_sd": (
                    float(np.std(ci_high_values, ddof=1))
                    if len(ci_high_values) > 1
                    else 0.0
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        args.output_dir / "bootstrap_summary.csv",
        index=False,
    )

    effect_summary_rows = []

    for (effect, metric), group in per_seed_effects.groupby(
        ["effect", "metric"], sort=True
    ):
        low = group["ci_low"].to_numpy(float)
        high = group["ci_high"].to_numpy(float)
        means = group["bootstrap_mean"].to_numpy(float)

        effect_summary_rows.append(
            {
                "effect": effect,
                "metric": metric,
                "n_seeds": len(group),
                "effect_mean": float(np.mean(means)),
                "effect_sd": (
                    float(np.std(means, ddof=1))
                    if len(means) > 1
                    else 0.0
                ),
                "bootstrap_ci_low_mean": float(np.mean(low)),
                "bootstrap_ci_high_mean": float(np.mean(high)),
            }
        )

    effect_summary_df = pd.DataFrame(effect_summary_rows)

    effect_summary_df.to_csv(
        args.output_dir / "bootstrap_effects_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # JSON for reproducibility
    # ------------------------------------------------------------------

    results_json = {
        "input": str(args.input),
        "n_bootstrap": args.n_bootstrap,
        "batch_size": args.batch_size,
        "confidence_level": args.confidence,
        "random_seed": args.random_seed,
        "seeds": [int(x) for x in seeds],
        "n_rows": int(len(df)),
        "n_plates_per_run": int(
            point_df.groupby("run_id")["n_plates"].first().iloc[0]
        ),
        "configs": list(CONFIGS),
        "bootstrap_method": {
            "unit": "plate",
            "replacement": True,
            "paired_comparisons": True,
            "ci_method": "percentile",
            "accuracy": "mean(correct)",
            "cer": "sum(edit_dist) / sum(gt_length)",
        },
        "factorial_effects": {
            "contrastive_effect_1d": "CTRBA - TRBA",
            "contrastive_effect_2d": "CTRB2A - TRB2A",
            "attention_effect_no_contrastive": "TRB2A - TRBA",
            "attention_effect_contrastive": "CTRB2A - CTRBA",
            "interaction": "(CTRB2A - TRB2A) - (CTRBA - TRBA)",
        },
    }

    with open(
        args.output_dir / "bootstrap_results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Human-readable console summary
    # ------------------------------------------------------------------

    print("")
    print("=" * 72)
    print("POINT ESTIMATES AND BOOTSTRAP CIs")
    print("=" * 72)

    for _, row in summary_df.sort_values(["metric", "config"]).iterrows():
        if row["metric"] == "accuracy":
            scale = 100.0
            suffix = "%"
        else:
            scale = 100.0
            suffix = "%"

        mean_value = row["point_mean"] * scale
        sd_value = row["point_sd"] * scale
        low_mean = row["bootstrap_ci_low_mean"] * scale
        high_mean = row["bootstrap_ci_high_mean"] * scale

        print(
            f"{row['config']:7s} {row['metric']:8s} "
            f"{mean_value:.4f}{suffix} ± {sd_value:.4f}{suffix} | "
            f"mean bootstrap CI: "
            f"[{low_mean:.4f}, {high_mean:.4f}]{suffix}"
        )

    print("")
    print("=" * 72)
    print("FACTORIAL EFFECTS")
    print("=" * 72)

    for _, row in effect_summary_df.sort_values(
        ["metric", "effect"]
    ).iterrows():
        scale = 100.0

        print(
            f"{row['effect']:35s} {row['metric']:8s} "
            f"{row['effect_mean'] * scale:+.4f}% "
            f"(seed SD={row['effect_sd'] * scale:.4f}%)"
        )

    print("")
    print("=" * 72)
    print("FILES")
    print("=" * 72)

    for filename in [
        "point_estimates.csv",
        "bootstrap_per_seed.csv",
        "bootstrap_summary.csv",
        "bootstrap_effects_per_seed.csv",
        "bootstrap_effects_summary.csv",
        "bootstrap_results.json",
    ]:
        print(f"  {args.output_dir / filename}")

    print("")
    print("[DONE]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        sys.exit(130)
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)