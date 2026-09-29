#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze bootstrap outputs produced by the ALPR bootstrap.py pipeline.

Compatible with Python 3.6.

Expected input directory:
    bootstrap_per_seed.csv
    bootstrap_summary.csv
    bootstrap_effects_per_seed.csv
    bootstrap_effects_summary.csv

Optional:
    point_estimates.csv
    bootstrap_results.json

Example:
    python analyze_bootstrap.py \
        --input-dir bootstrap_results/bootstrap \
        --output bootstrap_analysis.md
"""

from __future__ import print_function

import argparse
import json
import os
import sys

import pandas as pd


REQUIRED_FILES = [
    "bootstrap_per_seed.csv",
    "bootstrap_summary.csv",
    "bootstrap_effects_per_seed.csv",
    "bootstrap_effects_summary.csv",
]

OPTIONAL_FILES = [
    "point_estimates.csv",
    "bootstrap_results.json",
]


def fmt_pct(x, digits=2):
    if pd.isna(x):
        return "NA"
    return "{0:.{1}f}%".format(float(x) * 100.0, digits)


def fmt_pp(x, digits=2):
    if pd.isna(x):
        return "NA"
    return "{0:+.{1}f} p.p.".format(float(x) * 100.0, digits)


def fmt_num(x, digits=4):
    if pd.isna(x):
        return "NA"
    return "{0:.{1}f}".format(float(x), digits)


def load_inputs(input_dir):
    data = {}

    missing = []
    for name in REQUIRED_FILES:
        path = os.path.join(input_dir, name)
        if not os.path.isfile(path):
            missing.append(name)
        else:
            data[name] = pd.read_csv(path)

    if missing:
        raise RuntimeError(
            "Missing required files: " + ", ".join(missing)
        )

    for name in OPTIONAL_FILES:
        path = os.path.join(input_dir, name)
        if os.path.isfile(path):
            if name.endswith(".csv"):
                data[name] = pd.read_csv(path)
            else:
                with open(path, "r") as f:
                    data[name] = json.load(f)

    return data


def validate(data):
    problems = []
    warnings = []

    expected = {
        "bootstrap_per_seed.csv": [
            "seed", "config", "metric", "point_estimate",
            "bootstrap_mean", "bootstrap_sd",
            "ci_low", "ci_high", "n_bootstrap", "n_plates"
        ],
        "bootstrap_summary.csv": [
            "config", "metric", "n_seeds", "point_mean", "point_sd",
            "bootstrap_ci_low_mean", "bootstrap_ci_high_mean"
        ],
        "bootstrap_effects_per_seed.csv": [
            "seed", "effect", "metric", "bootstrap_mean",
            "bootstrap_sd", "ci_low", "ci_high",
            "n_bootstrap", "n_plates"
        ],
        "bootstrap_effects_summary.csv": [
            "effect", "metric", "n_seeds", "effect_mean",
            "effect_sd", "bootstrap_ci_low_mean",
            "bootstrap_ci_high_mean"
        ],
    }

    for name, cols in expected.items():
        missing = [c for c in cols if c not in data[name].columns]
        if missing:
            problems.append(
                "{}: missing columns: {}".format(name, ", ".join(missing))
            )

    if problems:
        return problems, warnings

    per_seed = data["bootstrap_per_seed.csv"]
    effects_per_seed = data["bootstrap_effects_per_seed.csv"]
    summary = data["bootstrap_summary.csv"]
    effects_summary = data["bootstrap_effects_summary.csv"]

    seeds = sorted(per_seed["seed"].unique().tolist())
    configs = sorted(per_seed["config"].unique().tolist())
    metrics = sorted(per_seed["metric"].unique().tolist())

    if len(seeds) != 5:
        warnings.append(
            "Expected 5 seeds, found {}: {}".format(len(seeds), seeds)
        )

    if set(configs) != set(["TRBA", "CTRBA", "TRB2A", "CTRB2A"]):
        warnings.append(
            "Unexpected configurations found: {}".format(configs)
        )

    if set(metrics) != set(["accuracy", "cer"]):
        warnings.append(
            "Unexpected metrics found: {}".format(metrics)
        )

    if "n_bootstrap" in per_seed.columns:
        vals = sorted(per_seed["n_bootstrap"].dropna().unique().tolist())
        if vals != [10000]:
            warnings.append(
                "n_bootstrap values are not exactly 10000: {}".format(vals)
            )

    if "n_plates" in per_seed.columns:
        vals = sorted(per_seed["n_plates"].dropna().unique().tolist())
        if vals != [9800]:
            warnings.append(
                "n_plates values are not exactly 9800: {}".format(vals)
            )

    if len(summary) != 8:
        warnings.append(
            "bootstrap_summary.csv has {} rows; expected 8.".format(len(summary))
        )

    if len(effects_summary) != 10:
        warnings.append(
            "bootstrap_effects_summary.csv has {} rows; expected 10.".format(
                len(effects_summary)
            )
        )

    # Verify the expected number of per-seed records.
    expected_metric_rows = len(seeds) * 4 * 2
    if len(per_seed) != expected_metric_rows:
        warnings.append(
            "bootstrap_per_seed.csv has {} rows; expected {}.".format(
                len(per_seed), expected_metric_rows
            )
        )

    expected_effect_rows = len(seeds) * 5 * 2
    if len(effects_per_seed) != expected_effect_rows:
        warnings.append(
            "bootstrap_effects_per_seed.csv has {} rows; expected {}.".format(
                len(effects_per_seed), expected_effect_rows
            )
        )

    return problems, warnings


def build_report(data, input_dir):
    per_seed = data["bootstrap_per_seed.csv"]
    summary = data["bootstrap_summary.csv"]
    effects_per_seed = data["bootstrap_effects_per_seed.csv"]
    effects_summary = data["bootstrap_effects_summary.csv"]

    lines = []

    lines.append("# ALPR Bootstrap Analysis")
    lines.append("")
    lines.append(
        "This report was generated automatically from the bootstrap output "
        "files in `{}`.".format(input_dir)
    )
    lines.append("")
    lines.append("> **Important:** bootstrap confidence intervals and "
                 "between-seed variability are reported separately. The "
                 "`*_mean` confidence-interval fields in the summary files "
                 "are the means of the per-seed CI limits; they are **not** "
                 "a single pooled 95% confidence interval across the five "
                 "training seeds.")
    lines.append("")

    # Methodology
    lines.append("## 1. Bootstrap procedure")
    lines.append("")
    json_data = data.get("bootstrap_results.json")
    if json_data:
        method = json_data.get("bootstrap_method", {})
        lines.append("- Bootstrap repetitions: **{}**".format(
            json_data.get("n_bootstrap", "NA")))
        lines.append("- Bootstrap unit: **{}**".format(
            method.get("unit", "NA")))
        lines.append("- Sampling: **with replacement = {}**".format(
            method.get("replacement", "NA")))
        lines.append("- Comparisons: **paired = {}**".format(
            method.get("paired_comparisons", "NA")))
        lines.append("- Confidence interval: **{} percentile method**".format(
            method.get("ci_method", "NA")))
        lines.append("- Accuracy: `{}`".format(
            method.get("accuracy", "NA")))
        lines.append("- CER: `{}`".format(
            method.get("cer", "NA")))
    else:
        lines.append(
            "The JSON metadata file was not available. The report uses the "
            "information contained in the four CSV files."
        )
    lines.append("")

    # Validation
    lines.append("## 2. Input validation")
    lines.append("")
    seeds = sorted(per_seed["seed"].unique().tolist())
    configs = sorted(per_seed["config"].unique().tolist())
    metrics = sorted(per_seed["metric"].unique().tolist())
    lines.append("- Seeds detected: **{}**".format(", ".join(map(str, seeds))))
    lines.append("- Configurations detected: **{}**".format(
        ", ".join(configs)))
    lines.append("- Metrics detected: **{}**".format(
        ", ".join(metrics)))
    lines.append("- Per-seed metric records: **{}**".format(len(per_seed)))
    lines.append("- Per-seed factorial-effect records: **{}**".format(
        len(effects_per_seed)))
    if "n_plates" in per_seed.columns:
        lines.append("- Plates per run: **{}**".format(
            int(per_seed["n_plates"].iloc[0])))
    if "n_bootstrap" in per_seed.columns:
        lines.append("- Bootstrap repetitions per run: **{}**".format(
            int(per_seed["n_bootstrap"].iloc[0])))
    lines.append("")

    # Performance table
    lines.append("## 3. Point estimates across the five seeds")
    lines.append("")
    lines.append(
        "| Configuration | Accuracy | Seed SD | CER | Seed SD |"
    )
    lines.append("|---|---:|---:|---:|---:|")

    order = ["TRBA", "CTRBA", "TRB2A", "CTRB2A"]
    for config in order:
        a = summary[
            (summary["config"] == config) &
            (summary["metric"] == "accuracy")
        ]
        c = summary[
            (summary["config"] == config) &
            (summary["metric"] == "cer")
        ]
        if len(a) == 1 and len(c) == 1:
            a = a.iloc[0]
            c = c.iloc[0]
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    config,
                    fmt_pct(a["point_mean"]),
                    fmt_pct(a["point_sd"]),
                    fmt_pct(c["point_mean"]),
                    fmt_pct(c["point_sd"]),
                )
            )
    lines.append("")
    lines.append(
        "The point estimate is the mean of the five independently trained "
        "models. The reported SD is the standard deviation across seeds."
    )
    lines.append("")

    # Per-seed consistency
    lines.append("## 4. Consistency across seeds")
    lines.append("")
    for config in order:
        lines.append("### {}".format(config))
        lines.append("")
        sub = per_seed[
            (per_seed["config"] == config) &
            (per_seed["metric"] == "accuracy")
        ].sort_values("seed")
        if len(sub):
            vals = []
            for _, row in sub.iterrows():
                vals.append(
                    "{}: {}".format(int(row["seed"]), fmt_pct(row["point_estimate"]))
                )
            lines.append("- Accuracy: " + "; ".join(vals))
        sub = per_seed[
            (per_seed["config"] == config) &
            (per_seed["metric"] == "cer")
        ].sort_values("seed")
        if len(sub):
            vals = []
            for _, row in sub.iterrows():
                vals.append(
                    "{}: {}".format(int(row["seed"]), fmt_pct(row["point_estimate"]))
                )
            lines.append("- CER: " + "; ".join(vals))
        lines.append("")

    # Per-seed CIs
    lines.append("## 5. Bootstrap confidence intervals by seed")
    lines.append("")
    lines.append(
        "The following intervals are the actual percentile bootstrap CIs "
        "reported for each individual seed."
    )
    lines.append("")

    lines.append("### Accuracy")
    lines.append("")
    lines.append("| Configuration | Seed | Estimate | 95% CI |")
    lines.append("|---|---:|---:|---:|")
    sub = per_seed[per_seed["metric"] == "accuracy"].copy()
    sub["config_order"] = sub["config"].map(
        {"TRBA": 0, "CTRBA": 1, "TRB2A": 2, "CTRB2A": 3}
    )
    sub = sub.sort_values(["config_order", "seed"])
    for _, row in sub.iterrows():
        lines.append(
            "| {} | {} | {} | [{}, {}] |".format(
                row["config"],
                int(row["seed"]),
                fmt_pct(row["point_estimate"]),
                fmt_pct(row["ci_low"]),
                fmt_pct(row["ci_high"]),
            )
        )
    lines.append("")

    lines.append("### CER")
    lines.append("")
    lines.append("| Configuration | Seed | Estimate | 95% CI |")
    lines.append("|---|---:|---:|---:|")
    sub = per_seed[per_seed["metric"] == "cer"].copy()
    sub["config_order"] = sub["config"].map(
        {"TRBA": 0, "CTRBA": 1, "TRB2A": 2, "CTRB2A": 3}
    )
    sub = sub.sort_values(["config_order", "seed"])
    for _, row in sub.iterrows():
        lines.append(
            "| {} | {} | {} | [{}, {}] |".format(
                row["config"],
                int(row["seed"]),
                fmt_pct(row["point_estimate"]),
                fmt_pct(row["ci_low"]),
                fmt_pct(row["ci_high"]),
            )
        )
    lines.append("")

    # Factorial effects
    lines.append("## 6. Factorial effects")
    lines.append("")
    lines.append(
        "| Effect | Accuracy | Seed SD | CER | Seed SD |"
    )
    lines.append("|---|---:|---:|---:|---:|")

    effect_order = [
        "contrastive_effect_1d",
        "contrastive_effect_2d",
        "attention_effect_no_contrastive",
        "attention_effect_contrastive",
        "interaction",
    ]

    for effect in effect_order:
        a = effects_summary[
            (effects_summary["effect"] == effect) &
            (effects_summary["metric"] == "accuracy")
        ]
        c = effects_summary[
            (effects_summary["effect"] == effect) &
            (effects_summary["metric"] == "cer")
        ]
        if len(a) == 1 and len(c) == 1:
            a = a.iloc[0]
            c = c.iloc[0]
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    effect,
                    fmt_pp(a["effect_mean"]),
                    fmt_pp(a["effect_sd"]),
                    fmt_pp(c["effect_mean"]),
                    fmt_pp(c["effect_sd"]),
                )
            )
    lines.append("")

    lines.append("### Interpretation of the factorial effects")
    lines.append("")
    lines.append(
        "- `contrastive_effect_1d = CTRBA - TRBA`: effect of Contrastive "
        "Learning when attention is 1D."
    )
    lines.append(
        "- `contrastive_effect_2d = CTRB2A - TRB2A`: effect of Contrastive "
        "Learning when attention is 2D."
    )
    lines.append(
        "- `attention_effect_no_contrastive = TRB2A - TRBA`: effect of "
        "2D attention without Contrastive Learning."
    )
    lines.append(
        "- `attention_effect_contrastive = CTRB2A - CTRBA`: effect of "
        "2D attention with Contrastive Learning."
    )
    lines.append(
        "- `interaction = (CTRB2A - TRB2A) - (CTRBA - TRBA)`: whether the "
        "effect of Contrastive Learning changes when moving from 1D to 2D "
        "attention."
    )
    lines.append("")

    # Effect CIs per seed
    lines.append("## 7. Bootstrap CIs for factorial effects by seed")
    lines.append("")
    lines.append(
        "These are per-seed bootstrap intervals for the paired factorial "
        "effects. They should not be interpreted as a pooled five-seed CI."
    )
    lines.append("")

    for metric in ["accuracy", "cer"]:
        lines.append("### {}".format(metric.upper()))
        lines.append("")
        lines.append("| Effect | Seed | Estimate | 95% CI | Contains zero? |")
        lines.append("|---|---:|---:|---:|:---:|")
        sub = effects_per_seed[
            effects_per_seed["metric"] == metric
        ].copy()
        sub["effect_order"] = sub["effect"].map(
            dict((name, i) for i, name in enumerate(effect_order))
        )
        sub = sub.sort_values(["effect_order", "seed"])
        for _, row in sub.iterrows():
            contains_zero = (
                float(row["ci_low"]) <= 0.0 <= float(row["ci_high"])
            )
            lines.append(
                "| {} | {} | {} | [{}, {}] | {} |".format(
                    row["effect"],
                    int(row["seed"]),
                    fmt_pp(row["bootstrap_mean"]),
                    fmt_pp(row["ci_low"]),
                    fmt_pp(row["ci_high"]),
                    "Yes" if contains_zero else "No",
                )
            )
        lines.append("")

    # Main findings
    lines.append("## 8. Main findings")
    lines.append("")

    # Pull effect means for prose.
    eff = {}
    for _, row in effects_summary.iterrows():
        eff[(row["effect"], row["metric"])] = row["effect_mean"]

    ctr1 = eff.get(("contrastive_effect_1d", "accuracy"))
    ctr2 = eff.get(("contrastive_effect_2d", "accuracy"))
    att0 = eff.get(("attention_effect_no_contrastive", "accuracy"))
    att1 = eff.get(("attention_effect_contrastive", "accuracy"))
    inter = eff.get(("interaction", "accuracy"))

    cer_ctr1 = eff.get(("contrastive_effect_1d", "cer"))
    cer_ctr2 = eff.get(("contrastive_effect_2d", "cer"))
    cer_att0 = eff.get(("attention_effect_no_contrastive", "cer"))
    cer_att1 = eff.get(("attention_effect_contrastive", "cer"))
    cer_inter = eff.get(("interaction", "cer"))

    if ctr1 is not None and ctr2 is not None:
        lines.append(
            "- The estimated Accuracy gain from Contrastive Learning is "
            "**{}** with 1D attention and **{}** with 2D attention.".format(
                fmt_pp(ctr1), fmt_pp(ctr2)
            )
        )
    if att0 is not None and att1 is not None:
        lines.append(
            "- The estimated Accuracy gain from 2D attention is **{}** "
            "without Contrastive Learning and **{}** with Contrastive "
            "Learning.".format(fmt_pp(att0), fmt_pp(att1))
        )
    if inter is not None:
        lines.append(
            "- The estimated Accuracy interaction is **{}**.".format(
                fmt_pp(inter)
            )
        )
    if cer_ctr1 is not None and cer_ctr2 is not None:
        lines.append(
            "- For CER, Contrastive Learning changes the estimated CER by "
            "**{}** with 1D attention and **{}** with 2D attention.".format(
                fmt_pp(cer_ctr1), fmt_pp(cer_ctr2)
            )
        )
    if cer_att0 is not None and cer_att1 is not None:
        lines.append(
            "- For CER, 2D attention changes the estimated CER by **{}** "
            "without Contrastive Learning and **{}** with Contrastive "
            "Learning.".format(fmt_pp(cer_att0), fmt_pp(cer_att1))
        )
    if cer_inter is not None:
        lines.append(
            "- The estimated CER interaction is **{}**.".format(
                fmt_pp(cer_inter)
            )
        )

    lines.append("")
    lines.append(
        "These statements describe estimated effects. Statistical claims "
        "about whether an effect is different from zero should be based on "
        "the per-seed bootstrap intervals and, if a single overall "
        "five-seed inferential statement is required, on a separately "
        "defined method for combining training-seed variability with "
        "test-sample uncertainty."
    )
    lines.append("")

    # Important limitation
    lines.append("## 9. Statistical interpretation and limitation")
    lines.append("")
    lines.append(
        "The current output separates two sources of variability: (1) "
        "variation between the five independent training seeds, summarized "
        "by the seed SD, and (2) test-set sampling uncertainty, summarized "
        "by the bootstrap CI for each seed."
    )
    lines.append("")
    lines.append(
        "The fields `bootstrap_ci_low_mean` and `bootstrap_ci_high_mean` "
        "in `bootstrap_summary.csv` and `bootstrap_effects_summary.csv` "
        "are averages of per-seed CI limits. They are useful as descriptive "
        "summaries, but they are not themselves a formally pooled 95% "
        "confidence interval across the five training runs."
    )
    lines.append("")
    lines.append(
        "Therefore, this report does not label an effect as statistically "
        "significant solely because the mean CI limits exclude zero. The "
        "per-seed bootstrap intervals are reported explicitly so that the "
        "robustness of the effect across training seeds can be assessed."
    )
    lines.append("")

    # Dissertation tables
    lines.append("## 10. Compact tables for the dissertation")
    lines.append("")
    lines.append("### Performance")
    lines.append("")
    lines.append("| Model | Accuracy (mean ± SD) | CER (mean ± SD) |")
    lines.append("|---|---:|---:|")
    for config in order:
        a = summary[
            (summary["config"] == config) &
            (summary["metric"] == "accuracy")
        ]
        c = summary[
            (summary["config"] == config) &
            (summary["metric"] == "cer")
        ]
        if len(a) == 1 and len(c) == 1:
            a = a.iloc[0]
            c = c.iloc[0]
            lines.append(
                "| {} | {} ± {} | {} ± {} |".format(
                    config,
                    fmt_pct(a["point_mean"]),
                    fmt_pct(a["point_sd"]),
                    fmt_pct(c["point_mean"]),
                    fmt_pct(c["point_sd"]),
                )
            )
    lines.append("")

    lines.append("### Factorial effects")
    lines.append("")
    lines.append("| Effect | Accuracy | CER |")
    lines.append("|---|---:|---:|")
    for effect in effect_order:
        a = effects_summary[
            (effects_summary["effect"] == effect) &
            (effects_summary["metric"] == "accuracy")
        ]
        c = effects_summary[
            (effects_summary["effect"] == effect) &
            (effects_summary["metric"] == "cer")
        ]
        if len(a) == 1 and len(c) == 1:
            a = a.iloc[0]
            c = c.iloc[0]
            lines.append(
                "| {} | {} | {} |".format(
                    effect,
                    fmt_pp(a["effect_mean"]),
                    fmt_pp(c["effect_mean"]),
                )
            )
    lines.append("")

    lines.append("## 11. Source files")
    lines.append("")
    for name in REQUIRED_FILES + OPTIONAL_FILES:
        if name in data:
            lines.append("- `{}`".format(name))
    lines.append("")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Markdown analysis from ALPR bootstrap CSV files."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing the bootstrap CSV files."
    )
    parser.add_argument(
        "--output",
        default="bootstrap_analysis.md",
        help="Output Markdown file."
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)

    if not os.path.isdir(input_dir):
        print("ERROR: input directory does not exist:", input_dir)
        return 1

    try:
        data = load_inputs(input_dir)
        problems, warnings = validate(data)

        if problems:
            print("ERROR: input validation failed:")
            for item in problems:
                print(" -", item)
            return 2

        report = build_report(data, input_dir)

        with open(args.output, "w") as f:
            f.write(report)

        print("=" * 72)
        print("ALPR Bootstrap Analysis")
        print("=" * 72)
        print("Input :", input_dir)
        print("Output:", os.path.abspath(args.output))

        if warnings:
            print("")
            print("WARNINGS:")
            for item in warnings:
                print(" -", item)

        print("")
        print("[DONE] Markdown report generated successfully.")
        return 0

    except Exception as exc:
        print("ERROR:", exc)
        return 3


if __name__ == "__main__":
    sys.exit(main())
