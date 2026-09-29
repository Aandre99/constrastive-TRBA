# ALPR Bootstrap Analysis

This report was generated automatically from the bootstrap output files in `/home/andre/dev/research/constrastive-deep-text-recognition-benchmark/bootstrap_results/bootstrap`.

> **Important:** bootstrap confidence intervals and between-seed variability are reported separately. The `*_mean` confidence-interval fields in the summary files are the means of the per-seed CI limits; they are **not** a single pooled 95% confidence interval across the five training seeds.

## 1. Bootstrap procedure

- Bootstrap repetitions: **10000**
- Bootstrap unit: **plate**
- Sampling: **with replacement = True**
- Comparisons: **paired = True**
- Confidence interval: **percentile percentile method**
- Accuracy: `mean(correct)`
- CER: `sum(edit_dist) / sum(gt_length)`

## 2. Input validation

- Seeds detected: **1111, 2222, 3333, 4444, 5555**
- Configurations detected: **CTRB2A, CTRBA, TRB2A, TRBA**
- Metrics detected: **accuracy, cer**
- Per-seed metric records: **40**
- Per-seed factorial-effect records: **50**
- Plates per run: **9800**
- Bootstrap repetitions per run: **10000**

## 3. Point estimates across the five seeds

| Configuration | Accuracy | Seed SD | CER | Seed SD |
|---|---:|---:|---:|---:|
| TRBA | 91.31% | 0.19% | 1.63% | 0.03% |
| CTRBA | 94.50% | 0.28% | 0.94% | 0.05% |
| TRB2A | 95.31% | 0.05% | 0.77% | 0.01% |
| CTRB2A | 97.18% | 0.26% | 0.45% | 0.03% |

The point estimate is the mean of the five independently trained models. The reported SD is the standard deviation across seeds.

## 4. Consistency across seeds

### TRBA

- Accuracy: 1111: 91.59%; 2222: 91.28%; 3333: 91.07%; 4444: 91.34%; 5555: 91.30%
- CER: 1111: 1.57%; 2222: 1.62%; 3333: 1.64%; 4444: 1.65%; 5555: 1.66%

### CTRBA

- Accuracy: 1111: 94.59%; 2222: 94.31%; 3333: 94.64%; 4444: 94.12%; 5555: 94.83%
- CER: 1111: 0.92%; 2222: 0.99%; 3333: 0.93%; 4444: 1.00%; 5555: 0.89%

### TRB2A

- Accuracy: 1111: 95.35%; 2222: 95.34%; 3333: 95.23%; 4444: 95.31%; 5555: 95.34%
- CER: 1111: 0.77%; 2222: 0.76%; 3333: 0.79%; 4444: 0.77%; 5555: 0.76%

### CTRB2A

- Accuracy: 1111: 96.84%; 2222: 97.17%; 3333: 97.52%; 4444: 97.33%; 5555: 97.04%
- CER: 1111: 0.49%; 2222: 0.46%; 3333: 0.41%; 4444: 0.42%; 5555: 0.47%

## 5. Bootstrap confidence intervals by seed

The following intervals are the actual percentile bootstrap CIs reported for each individual seed.

### Accuracy

| Configuration | Seed | Estimate | 95% CI |
|---|---:|---:|---:|
| TRBA | 1111 | 91.59% | [91.05%, 92.13%] |
| TRBA | 2222 | 91.28% | [90.71%, 91.84%] |
| TRBA | 3333 | 91.07% | [90.51%, 91.63%] |
| TRBA | 4444 | 91.34% | [90.79%, 91.89%] |
| TRBA | 5555 | 91.30% | [90.71%, 91.85%] |
| CTRBA | 1111 | 94.59% | [94.14%, 95.03%] |
| CTRBA | 2222 | 94.31% | [93.85%, 94.76%] |
| CTRBA | 3333 | 94.64% | [94.19%, 95.09%] |
| CTRBA | 4444 | 94.12% | [93.67%, 94.58%] |
| CTRBA | 5555 | 94.83% | [94.39%, 95.27%] |
| TRB2A | 1111 | 95.35% | [94.93%, 95.77%] |
| TRB2A | 2222 | 95.34% | [94.92%, 95.73%] |
| TRB2A | 3333 | 95.23% | [94.81%, 95.64%] |
| TRB2A | 4444 | 95.31% | [94.89%, 95.72%] |
| TRB2A | 5555 | 95.34% | [94.91%, 95.76%] |
| CTRB2A | 1111 | 96.84% | [96.49%, 97.18%] |
| CTRB2A | 2222 | 97.17% | [96.83%, 97.49%] |
| CTRB2A | 3333 | 97.52% | [97.21%, 97.83%] |
| CTRB2A | 4444 | 97.33% | [97.01%, 97.64%] |
| CTRB2A | 5555 | 97.04% | [96.70%, 97.38%] |

### CER

| Configuration | Seed | Estimate | 95% CI |
|---|---:|---:|---:|
| TRBA | 1111 | 1.57% | [1.46%, 1.68%] |
| TRBA | 2222 | 1.62% | [1.51%, 1.73%] |
| TRBA | 3333 | 1.64% | [1.53%, 1.75%] |
| TRBA | 4444 | 1.65% | [1.54%, 1.77%] |
| TRBA | 5555 | 1.66% | [1.54%, 1.78%] |
| CTRBA | 1111 | 0.92% | [0.84%, 1.00%] |
| CTRBA | 2222 | 0.99% | [0.90%, 1.07%] |
| CTRBA | 3333 | 0.93% | [0.85%, 1.02%] |
| CTRBA | 4444 | 1.00% | [0.91%, 1.08%] |
| CTRBA | 5555 | 0.89% | [0.81%, 0.97%] |
| TRB2A | 1111 | 0.77% | [0.70%, 0.84%] |
| TRB2A | 2222 | 0.76% | [0.69%, 0.84%] |
| TRB2A | 3333 | 0.79% | [0.72%, 0.86%] |
| TRB2A | 4444 | 0.77% | [0.70%, 0.85%] |
| TRB2A | 5555 | 0.76% | [0.69%, 0.84%] |
| CTRB2A | 1111 | 0.49% | [0.44%, 0.55%] |
| CTRB2A | 2222 | 0.46% | [0.40%, 0.51%] |
| CTRB2A | 3333 | 0.41% | [0.36%, 0.47%] |
| CTRB2A | 4444 | 0.42% | [0.37%, 0.47%] |
| CTRB2A | 5555 | 0.47% | [0.41%, 0.52%] |

## 6. Factorial effects

| Effect | Accuracy | Seed SD | CER | Seed SD |
|---|---:|---:|---:|---:|
| contrastive_effect_1d | +3.18 p.p. | +0.35 p.p. | -0.68 p.p. | +0.06 p.p. |
| contrastive_effect_2d | +1.87 p.p. | +0.30 p.p. | -0.32 p.p. | +0.04 p.p. |
| attention_effect_no_contrastive | +4.00 p.p. | +0.15 p.p. | -0.86 p.p. | +0.04 p.p. |
| attention_effect_contrastive | +2.68 p.p. | +0.43 p.p. | -0.50 p.p. | +0.07 p.p. |
| interaction | -1.32 p.p. | +0.39 p.p. | +0.36 p.p. | +0.07 p.p. |

### Interpretation of the factorial effects

- `contrastive_effect_1d = CTRBA - TRBA`: effect of Contrastive Learning when attention is 1D.
- `contrastive_effect_2d = CTRB2A - TRB2A`: effect of Contrastive Learning when attention is 2D.
- `attention_effect_no_contrastive = TRB2A - TRBA`: effect of 2D attention without Contrastive Learning.
- `attention_effect_contrastive = CTRB2A - CTRBA`: effect of 2D attention with Contrastive Learning.
- `interaction = (CTRB2A - TRB2A) - (CTRBA - TRBA)`: whether the effect of Contrastive Learning changes when moving from 1D to 2D attention.

## 7. Bootstrap CIs for factorial effects by seed

These are per-seed bootstrap intervals for the paired factorial effects. They should not be interpreted as a pooled five-seed CI.

### ACCURACY

| Effect | Seed | Estimate | 95% CI | Contains zero? |
|---|---:|---:|---:|:---:|
| contrastive_effect_1d | 1111 | +3.00 p.p. | [+2.55 p.p., +3.45 p.p.] | No |
| contrastive_effect_1d | 2222 | +3.03 p.p. | [+2.57 p.p., +3.51 p.p.] | No |
| contrastive_effect_1d | 3333 | +3.57 p.p. | [+3.09 p.p., +4.05 p.p.] | No |
| contrastive_effect_1d | 4444 | +2.79 p.p. | [+2.35 p.p., +3.23 p.p.] | No |
| contrastive_effect_1d | 5555 | +3.53 p.p. | [+3.08 p.p., +3.99 p.p.] | No |
| contrastive_effect_2d | 1111 | +1.49 p.p. | [+1.12 p.p., +1.86 p.p.] | No |
| contrastive_effect_2d | 2222 | +1.84 p.p. | [+1.48 p.p., +2.20 p.p.] | No |
| contrastive_effect_2d | 3333 | +2.29 p.p. | [+1.91 p.p., +2.66 p.p.] | No |
| contrastive_effect_2d | 4444 | +2.01 p.p. | [+1.64 p.p., +2.39 p.p.] | No |
| contrastive_effect_2d | 5555 | +1.70 p.p. | [+1.35 p.p., +2.06 p.p.] | No |
| attention_effect_no_contrastive | 1111 | +3.76 p.p. | [+3.23 p.p., +4.30 p.p.] | No |
| attention_effect_no_contrastive | 2222 | +4.06 p.p. | [+3.51 p.p., +4.60 p.p.] | No |
| attention_effect_no_contrastive | 3333 | +4.16 p.p. | [+3.62 p.p., +4.70 p.p.] | No |
| attention_effect_no_contrastive | 4444 | +3.97 p.p. | [+3.46 p.p., +4.48 p.p.] | No |
| attention_effect_no_contrastive | 5555 | +4.04 p.p. | [+3.51 p.p., +4.58 p.p.] | No |
| attention_effect_contrastive | 1111 | +2.25 p.p. | [+1.80 p.p., +2.70 p.p.] | No |
| attention_effect_contrastive | 2222 | +2.87 p.p. | [+2.42 p.p., +3.32 p.p.] | No |
| attention_effect_contrastive | 3333 | +2.88 p.p. | [+2.44 p.p., +3.33 p.p.] | No |
| attention_effect_contrastive | 4444 | +3.20 p.p. | [+2.73 p.p., +3.64 p.p.] | No |
| attention_effect_contrastive | 5555 | +2.22 p.p. | [+1.78 p.p., +2.65 p.p.] | No |
| interaction | 1111 | -1.51 p.p. | [-2.06 p.p., -0.95 p.p.] | No |
| interaction | 2222 | -1.20 p.p. | [-1.77 p.p., -0.61 p.p.] | No |
| interaction | 3333 | -1.28 p.p. | [-1.87 p.p., -0.68 p.p.] | No |
| interaction | 4444 | -0.77 p.p. | [-1.33 p.p., -0.22 p.p.] | No |
| interaction | 5555 | -1.83 p.p. | [-2.40 p.p., -1.28 p.p.] | No |

### CER

| Effect | Seed | Estimate | 95% CI | Contains zero? |
|---|---:|---:|---:|:---:|
| contrastive_effect_1d | 1111 | -0.65 p.p. | [-0.73 p.p., -0.57 p.p.] | No |
| contrastive_effect_1d | 2222 | -0.63 p.p. | [-0.72 p.p., -0.55 p.p.] | No |
| contrastive_effect_1d | 3333 | -0.71 p.p. | [-0.79 p.p., -0.62 p.p.] | No |
| contrastive_effect_1d | 4444 | -0.65 p.p. | [-0.74 p.p., -0.57 p.p.] | No |
| contrastive_effect_1d | 5555 | -0.77 p.p. | [-0.86 p.p., -0.69 p.p.] | No |
| contrastive_effect_2d | 1111 | -0.28 p.p. | [-0.34 p.p., -0.21 p.p.] | No |
| contrastive_effect_2d | 2222 | -0.31 p.p. | [-0.37 p.p., -0.24 p.p.] | No |
| contrastive_effect_2d | 3333 | -0.38 p.p. | [-0.44 p.p., -0.32 p.p.] | No |
| contrastive_effect_2d | 4444 | -0.35 p.p. | [-0.41 p.p., -0.29 p.p.] | No |
| contrastive_effect_2d | 5555 | -0.29 p.p. | [-0.36 p.p., -0.24 p.p.] | No |
| attention_effect_no_contrastive | 1111 | -0.80 p.p. | [-0.91 p.p., -0.70 p.p.] | No |
| attention_effect_no_contrastive | 2222 | -0.86 p.p. | [-0.96 p.p., -0.75 p.p.] | No |
| attention_effect_no_contrastive | 3333 | -0.85 p.p. | [-0.95 p.p., -0.75 p.p.] | No |
| attention_effect_no_contrastive | 4444 | -0.88 p.p. | [-0.98 p.p., -0.78 p.p.] | No |
| attention_effect_no_contrastive | 5555 | -0.90 p.p. | [-1.01 p.p., -0.80 p.p.] | No |
| attention_effect_contrastive | 1111 | -0.43 p.p. | [-0.51 p.p., -0.35 p.p.] | No |
| attention_effect_contrastive | 2222 | -0.53 p.p. | [-0.61 p.p., -0.45 p.p.] | No |
| attention_effect_contrastive | 3333 | -0.52 p.p. | [-0.60 p.p., -0.45 p.p.] | No |
| attention_effect_contrastive | 4444 | -0.57 p.p. | [-0.65 p.p., -0.49 p.p.] | No |
| attention_effect_contrastive | 5555 | -0.42 p.p. | [-0.50 p.p., -0.35 p.p.] | No |
| interaction | 1111 | +0.38 p.p. | [+0.28 p.p., +0.47 p.p.] | No |
| interaction | 2222 | +0.33 p.p. | [+0.22 p.p., +0.42 p.p.] | No |
| interaction | 3333 | +0.33 p.p. | [+0.23 p.p., +0.43 p.p.] | No |
| interaction | 4444 | +0.30 p.p. | [+0.21 p.p., +0.40 p.p.] | No |
| interaction | 5555 | +0.48 p.p. | [+0.38 p.p., +0.58 p.p.] | No |

## 8. Main findings

- The estimated Accuracy gain from Contrastive Learning is **+3.18 p.p.** with 1D attention and **+1.87 p.p.** with 2D attention.
- The estimated Accuracy gain from 2D attention is **+4.00 p.p.** without Contrastive Learning and **+2.68 p.p.** with Contrastive Learning.
- The estimated Accuracy interaction is **-1.32 p.p.**.
- For CER, Contrastive Learning changes the estimated CER by **-0.68 p.p.** with 1D attention and **-0.32 p.p.** with 2D attention.
- For CER, 2D attention changes the estimated CER by **-0.86 p.p.** without Contrastive Learning and **-0.50 p.p.** with Contrastive Learning.
- The estimated CER interaction is **+0.36 p.p.**.

These statements describe estimated effects. Statistical claims about whether an effect is different from zero should be based on the per-seed bootstrap intervals and, if a single overall five-seed inferential statement is required, on a separately defined method for combining training-seed variability with test-sample uncertainty.

## 9. Statistical interpretation and limitation

The current output separates two sources of variability: (1) variation between the five independent training seeds, summarized by the seed SD, and (2) test-set sampling uncertainty, summarized by the bootstrap CI for each seed.

The fields `bootstrap_ci_low_mean` and `bootstrap_ci_high_mean` in `bootstrap_summary.csv` and `bootstrap_effects_summary.csv` are averages of per-seed CI limits. They are useful as descriptive summaries, but they are not themselves a formally pooled 95% confidence interval across the five training runs.

Therefore, this report does not label an effect as statistically significant solely because the mean CI limits exclude zero. The per-seed bootstrap intervals are reported explicitly so that the robustness of the effect across training seeds can be assessed.

## 10. Compact tables for the dissertation

### Performance

| Model | Accuracy (mean ± SD) | CER (mean ± SD) |
|---|---:|---:|
| TRBA | 91.31% ± 0.19% | 1.63% ± 0.03% |
| CTRBA | 94.50% ± 0.28% | 0.94% ± 0.05% |
| TRB2A | 95.31% ± 0.05% | 0.77% ± 0.01% |
| CTRB2A | 97.18% ± 0.26% | 0.45% ± 0.03% |

### Factorial effects

| Effect | Accuracy | CER |
|---|---:|---:|
| contrastive_effect_1d | +3.18 p.p. | -0.68 p.p. |
| contrastive_effect_2d | +1.87 p.p. | -0.32 p.p. |
| attention_effect_no_contrastive | +4.00 p.p. | -0.86 p.p. |
| attention_effect_contrastive | +2.68 p.p. | -0.50 p.p. |
| interaction | -1.32 p.p. | +0.36 p.p. |

## 11. Source files

- `bootstrap_per_seed.csv`
- `bootstrap_summary.csv`
- `bootstrap_effects_per_seed.csv`
- `bootstrap_effects_summary.csv`
- `point_estimates.csv`
- `bootstrap_results.json`

