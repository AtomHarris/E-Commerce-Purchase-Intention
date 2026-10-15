# E-Commerce Purchase Intention Prediction

Predicting whether an online shopping session ends in a purchase, and turning those predictions into cost-aware business decisions.

Submitted for **DS-7003 Advanced Decision Making: Predictive Analytics & Machine Learning** (UNICAF), Week 12 Summative Assignment.

**Author:** Harris Mugambi Lukundi

---

## Overview

Online retailers collect large amounts of clickstream data but often struggle to act on it in time. This project analyses **12,330 browsing sessions** to:

1. identify the behavioural patterns that distinguish purchasing sessions from non-purchasing ones, and
2. build and compare classifiers that predict purchase intent, then show how the predicted scores should be converted into decisions using the cost and benefit of an intervention rather than the default 0.5 threshold.

The work follows the CRISP-DM methodology. The full write-up is in [`report/Harris_R2411D18873136_Report.pdf`](report/Harris_R2411D18873136_Report.pdf).

## Dataset

[Online Shoppers Purchasing Intention Dataset](https://doi.org/10.24432/C5F88Q) (Sakar and Kastro, 2018), UCI Machine Learning Repository, licensed CC BY 4.0.

- 12,330 sessions, 18 attributes (10 numerical, 8 categorical), no missing values
- Target: `Revenue` (purchase / no purchase), with only **15.5%** positive, so the classes are heavily imbalanced
- Expanded to 28 model inputs after one-hot encoding `Month` and `VisitorType`

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   └── online_shoppers_intention.csv
├── notebooks/
│   └── purchase_intention_analysis.ipynb
└── report/
    └── Harris_R2411D18873136_Report.pdf
```

## Methodology

| Stage | Details |
|---|---|
| Preparation | Boolean fields cast to integers; `Month` and `VisitorType` one-hot encoded; 75/25 stratified train/test split (`random_state=22`) |
| Imbalance | Class weighting (`class_weight="balanced"`, or `sample_weight` of 5.5 for purchasers where the model has no such option). Preferred to SMOTE to avoid synthetic customers |
| Models | Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, Naive Bayes, Multilayer Perceptron (default settings, not tuned) |
| Evaluation | Accuracy, precision, recall, F1 and AUC on the test set; 5-fold stratified cross-validation; confusion matrices; ROC and precision-recall curves |
| Interpretation | Impurity-based and permutation feature importance; threshold analysis |
| Decision analysis | Expected-utility threshold `p* = C / (B + C)`, adjusted for the class weighting used in training |

## Key results

Test-set performance at the default 0.5 threshold (precision, recall and F1 refer to purchases):

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| **Gradient Boosting** | 0.8693 | 0.5499 | 0.8553 | 0.6694 | 0.9345 |
| Random Forest | 0.9046 | 0.7668 | 0.5514 | 0.6415 | 0.9276 |
| AdaBoost | 0.8647 | 0.5418 | 0.8155 | 0.6510 | 0.9177 |
| Logistic Regression | 0.8586 | 0.5304 | 0.7505 | 0.6215 | 0.9033 |
| MLP | 0.8884 | 0.6572 | 0.5828 | 0.6178 | 0.8951 |
| Decision Tree | 0.8440 | 0.4973 | 0.7652 | 0.6028 | 0.8515 |
| Naive Bayes | 0.7139 | 0.3130 | 0.7107 | 0.4346 | 0.8026 |

Main findings:

- **Gradient Boosting** performed best on the test split, but five-fold cross-validation shows it ranks sessions no better than Random Forest (mean AUC 0.9322 vs 0.9306, difference smaller than its fold-to-fold variation). The two differ mainly in operating point: Gradient Boosting favours recall, Random Forest favours precision.
- **Page Value** is by far the strongest predictor (about 77% of impurity-based importance).
- **Seasonality:** conversion peaks in November (25.4%) and is lowest in February (1.6%).
- **Visitor type:** new visitors convert at 24.9%, returning visitors at 13.9%.
- **Thresholds should reflect cost.** For a cheap intervention (B = £40, C = £4) the score cut-off is about 0.35; for a costly one (C = £20) about 0.73. Scores from the class-weighted model are not true probabilities, so any cut-off should be piloted first.

## Limitations

- Single dataset from one organisation, covering ten months (January and April are missing).
- Models were not hyperparameter-tuned.
- Page Value is derived from historical page revenue and may partly reflect the purchase outcome (possible data leakage), so results may be optimistic. Re-testing without it is recommended.
- Class-weighted scores are not calibrated probabilities, so threshold values are approximate.

## Getting started

**Requirements:** Python 3.9+

```bash
# 1. Clone the repository
git clone https://github.com/AtomHarris/E-Commerce-Purchase-Intention.git
cd E-Commerce-Purchase-Intention

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook notebooks/purchase_intention_analysis.ipynb
```

If `data/online_shoppers_intention.csv` is missing, download it from the [UCI repository](https://doi.org/10.24432/C5F88Q) and place it in `data/`.

## References

- Sakar, C. and Kastro, Y. (2018) *Online shoppers purchasing intention dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q
- Sakar, C.O., Polat, S.O., Katircioglu, M. and Kastro, Y. (2019) 'Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks', *Neural Computing and Applications*, 31(10), pp. 6893-6908.
- Chapman, P. et al. (2000) *CRISP-DM 1.0: step-by-step data mining guide*. SPSS Inc.

The full reference list is in the report.

## License

Code is released under the [MIT License](LICENSE). The dataset is © its original authors and licensed CC BY 4.0.
