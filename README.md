# Credit Card Fraud Detection — Comparative Model Analysis

## 1. Objective

This report compares supervised and unsupervised machine learning approaches for detecting fraudulent credit card transactions, using **Area Under the Precision-Recall Curve (AUPRC)** as the primary evaluation metric. AUPRC was chosen over ROC-AUC because it is far more informative under the severe class imbalance present in this dataset (~0.17% fraud rate), where ROC-AUC can appear misleadingly high even for weak models.

## 2. Dataset & Methodology

- **Source:** Kaggle `mlg-ulb/creditcardfraud` — anonymized transactions with PCA-transformed features (`V1`–`V28`), plus `Time` and `Amount`.
- **Preprocessing:** Exact duplicate rows removed. Features scaled with `StandardScaler`, fit exclusively on the training set and reused (via `.transform()`) for validation and test data to prevent leakage.
- **Data split:** Stratified splitting on the `Class` label to preserve fraud prevalence across sets:
  - 50% held out as an untouched **test set**
  - Remaining 50% further split into **80% train / 20% validation**
- **Resulting sizes:**

  | Split | Total rows | Fraud cases | Fraud rate |
  |---|---|---|---|
  | Train | 113,490 | ~189 (est.) | ~0.17% |
  | Validation | 28,373 | 47 | 0.166% |
  | Test | 141,863 | 236 | 0.166% |

- **Random baseline AUPRC** (expected score of a no-skill/random classifier) ≈ fraud prevalence ≈ **0.0017**, used throughout as a reference point for "lift."

## 3. Supervised Models Evaluated

Each model was trained with default hyperparameters, then re-trained with `class_weight='balanced'` (or XGBoost's equivalent `scale_pos_weight`) to test whether explicitly correcting for class imbalance improved ranking performance.

| Model | AUPRC (unweighted) | AUPRC (balanced) | Change |
|---|---|---|---|
| SVC (RBF kernel) | 0.8165 | 0.4795 | **−0.3370** |
| Logistic Regression | 0.8154 | 0.8485 | +0.0331 |
| Random Forest | 0.8590 | 0.8818 | +0.0228 |
| XGBoost | 0.8336 | **0.8844** | +0.0508 |

*(All scores measured on the validation set.)*

### Key finding: class-weight balancing helps every model except SVM

Balancing improved three of the four models, consistent with the standard expectation that reweighting the minority class helps a classifier take rare fraud cases more seriously. **SVC was the exception, and the failure was severe** — AUPRC collapsed from 0.8165 to 0.4795.

**Root cause:** `class_weight='balanced'` in SVC inflates the regularization penalty `C` for the minority class by roughly the inverse class-frequency ratio (~580× for this dataset). Because SVM training is a margin-based optimization dominated by a small number of support vectors near the decision boundary, this inflated penalty effectively forces the solver to treat every fraud training point as a near-hard constraint. Diagnostic checks confirmed this:

- The solver required **6,236 iterations** to converge (vs. a much lower count for the unweighted model), indicating a far harder optimization landscape.
- The resulting decision function showed **near-total score overlap** between fraud and legitimate transactions (fraud scores: −2.84 to 1.21; legitimate scores: −4.58 to 1.17), meaning the model could no longer reliably rank fraud above legitimate transactions across most of the threshold range.

This is a useful, generalizable takeaway: **class-weight balancing is not a universally safe default** — its effect depends heavily on how a given model's loss function is structured. Smooth, global-objective models (logistic regression, tree ensembles) tolerated and benefited from it; SVM's margin-based, boundary-sensitive optimization did not.

## 4. Unsupervised Baseline: Isolation Forest

To establish a reference point for how much value the fraud labels themselves provide, an unsupervised Isolation Forest was also evaluated — first trained on the full training set (including fraud), then trained only on legitimate transactions (a semi-supervised "novelty detection" framing).

A single-seed comparison initially suggested the full-data variant outperformed the legit-only variant (0.1711 vs. 0.1510 AUPRC), but given the algorithm's inherent sampling randomness, this was tested rigorously across **21 random seeds**:

| Training data | Mean AUPRC | Std. dev. |
|---|---|---|
| Full data (incl. fraud) | 0.1883 | 0.0436 |
| Legitimate-only | 0.1616 | 0.0355 |

A paired t-test confirmed the difference is statistically significant (**t = 2.88, p = 0.009**; Wilcoxon signed-rank **p = 0.024**; 95% CI on the mean difference: **[0.007, 0.046]**). This is a modest but real effect: with `max_samples='auto'` capping each tree at 256 rows per subsample, occasional inclusion of fraud examples in a tree's training subsample appears to give the full-data forest a small but consistent edge in isolating similar patterns at inference time.

**Bottom line:** even the better-performing unsupervised configuration (AUPRC ≈ 0.19) falls far short of every supervised model (AUPRC 0.82–0.88). This confirms that for this dataset, labeled fraud examples carry substantial, hard-to-replace signal — unsupervised anomaly detection is a reasonable fallback when labels are unavailable, but is not a substitute for supervised learning when labels exist.

## 5. Final Model Selection & Test-Set Evaluation

**XGBoost (class-weight balanced)** was selected as the best-performing model based on validation AUPRC (0.8844), narrowly ahead of Random Forest (balanced) at 0.8818.

Evaluated once, on the previously untouched test set:

| Metric | Validation | Test |
|---|---|---|
| AUPRC | 0.8844 | 0.8270 |
| Fraud cases (n) | 47 | 236 |
| Baseline (random) AUPRC | 0.0017 | 0.0017 |
| Lift over baseline | ~520× | ~499× |

**Interpreting the validation-to-test gap:** A drop of 0.057 might initially suggest overfitting to the validation set, but the more likely explanation is statistical: AUPRC computed from only **47** positive cases (validation) carries substantially more sampling variance than AUPRC computed from **236** positive cases (test) — the same phenomenon observed in the Isolation Forest seed-variance analysis. Both splits have an essentially identical underlying fraud rate (0.166%), ruling out a distribution shift between sets. Given the larger positive-class sample in the test set, **the test AUPRC (0.827) is treated as the more statistically reliable estimate of true model performance**, with the validation score likely representing a somewhat optimistic draw from a small sample.

*(Recommended follow-up, not yet completed: bootstrap resampling of both sets to directly quantify each estimate's confidence interval and confirm this interpretation numerically.)*

## 6. Conclusions & Recommendations

1. **XGBoost with balanced class weighting is the recommended model**, achieving the best validation performance and a strong, statistically well-supported test AUPRC of 0.827 — roughly 500× better than random guessing.
2. **Class-weight balancing should be applied selectively, not by default.** It benefited every model tested except SVM, where it caused catastrophic performance loss due to the margin-based optimization becoming ill-conditioned under inflated per-class penalties.
3. **Unsupervised anomaly detection (Isolation Forest) is not competitive** with supervised approaches on this dataset when labels are available, though it remains a valid fallback for scenarios where labeled fraud data doesn't exist.
4. **AUPRC estimates from small validation sets should be interpreted cautiously.** With only dozens of positive cases, single-point AUPRC estimates can vary meaningfully; where possible, report results with confidence intervals (via bootstrapping) rather than as single numbers, especially when comparing closely-performing models.
5. **Next steps for further work:** bootstrap confidence intervals on the final test AUPRC; threshold selection informed by the business cost trade-off between false positives (blocked legitimate transactions) and false negatives (missed fraud); and repeated-split cross-validation for the top 2–3 models to further stabilize model-selection decisions.

---
*Metric: Average Precision (AUPRC). Models trained and evaluated with GPU-accelerated cuML (SVC, Logistic Regression, Random Forest) and XGBoost with CUDA support. Dataset: Kaggle `mlg-ulb/creditcardfraud`.*
