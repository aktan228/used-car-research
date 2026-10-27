# Machine Learning Practical Assignments 1–7

**Student:** Aktan Iygilikovich Kashymbekov

**Institution:** International College of IT and Business, Salymbekov University

**Assignment level:** C

**Environment:** Python 3.12.10, VS Code, Jupyter, Windows

The main project explores used-car characteristics and predicts asking prices. Practice 5 identifies diesel cars in an imbalanced dataset. Practice 6 uses a separate corpus of Kyrgyz news for topic classification.

The repository directory, `ml-social-media-sleep`, retains its name from an earlier topic. The seven current notebooks cover used cars and news.

## Repository structure

Each practical assignment has its own directory. The car dataset is shared by practices 1–5 and 7; the news CSVs belong to practice 6. Each practice README describes its objective, workflow, recorded results, and execution instructions.

```text
ml-social-media-sleep/
├── README.md
├── requirements.txt
├── data/
│   └── raw/
│       └── used_cars.csv
├── practice1/
│   ├── README.md
│   └── 01_used_car_analysis.ipynb
├── practice2/
│   ├── README.md
│   └── 02_EDA_of_used_cars.ipynb
├── practice3/
│   ├── README.md
│   └── 03_used_cars_pipeline.ipynb
├── practice4/
│   ├── README.md
│   └── 04_used_cars_regression.ipynb
├── practice5/
│   ├── README.md
│   └── 05_used_cars_classification.ipynb
├── practice6/
│   ├── README.md
│   ├── 06_kyrgyz_news_classification.ipynb
│   └── data/
│       ├── qad-train_1000.csv
│       └── qad-test_0500.csv
└── practice7/
    ├── README.md
    └── 07_used_cars_random_forest.ipynb
```

`.venv/` is the local virtual environment. Each notebook contains its own required function and class definitions and can be run in a fresh kernel without first running another notebook.

## Datasets

### Used cars: practices 1–5 and 7

**Used Car Price Prediction Dataset**, published by Taeef Najib. Source: [Kaggle](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset). Local file: [used_cars.csv](data/raw/used_cars.csv).

The dataset contains **4,009 listings and 12 columns**. Each row represents a car listing. The regression target, `price`, is the asking price in US dollars (USD), which may differ from the final sale price.

| Column | Meaning | Semantic type |
|---|---|---|
| `brand` | Car manufacturer | Categorical |
| `model` | Model and trim | Categorical |
| `model_year` | Model year | Numeric |
| `milage` | Mileage in miles; the original column spelling is retained | Numeric, stored as text |
| `fuel_type` | Fuel type | Categorical |
| `engine` | Engine description | Text |
| `transmission` | Transmission description | Categorical |
| `ext_col` | Exterior color | Categorical |
| `int_col` | Interior color | Categorical |
| `accident` | Reported accidents or damage | Categorical |
| `clean_title` | Clean-title status reported in the listing | Categorical |
| `price` | Asking price in USD | Numeric, stored as text |

There are no exact duplicate rows. Missing values: `fuel_type` — 170, `accident` — 113, `clean_title` — 596. Fuel values also include `–` and `not supported`. No reported accident does not guarantee an accident-free history; a missing `clean_title` value does not mean “No.”

The combination of numeric and categorical features, text-formatted prices and mileage, missing values, and inconsistent descriptions supports analysis, preprocessing, and modeling across the assignments.

### Kyrgyz news: practice 6

Sources: [Kyrgyz Multi-Label Topic Classification](https://github.com/alexeyev/kyrgyz-multi-label-topic-classification) and the [authors' paper](https://arxiv.org/pdf/2308.15952). The corpus contains Kyrgyz-language news from 24.kg with topic annotations.

Local files: [qad-train_1000.csv](practice6/data/qad-train_1000.csv), with 1,000 rows, and [qad-test_0500.csv](practice6/data/qad-test_0500.csv), with 500 rows. Columns: `id`, `text`, `labels_set`, `tokenized_text`.

The original annotations allow multiple topics per article. For this assignment, only single-label records are retained. Empty texts and normalized duplicates are removed, and classes with fewer than 20 documents are excluded. The resulting sample contains **696 documents across 10 classes**.

The original files are combined before creating a new stratified split: **556 training and 140 test documents**. Reported results apply to this educational sample and are not directly comparable to the authors' original benchmark.

## Assignment overview

| No. | Guide | Notebook | Implemented work |
|---|---|---|---|
| 1 | [README](practice1/README.md) | [Initial analysis](practice1/01_used_car_analysis.ipynb) | Loading, dimensions, types, missing values, duplicates, categories, numeric price parsing, and a dictionary of 12 features |
| 2 | [README](practice2/README.md) | [Exploratory data analysis](practice2/02_EDA_of_used_cars.ipynb) | Histograms, scatter plots, boxplots, heatmap, three statistical hypotheses, Bonferroni correction, data issues, and leakage risks |
| 3 | [README](practice3/README.md) | [Transformer and pipeline](practice3/03_used_cars_pipeline.ipynb) | `BrandMileageTransformer`, relative mileage by brand, imputation, scaling, one-hot encoding, three tests, and performance measurements |
| 4 | [README](practice4/README.md) | [Regression](practice4/04_used_cars_regression.ipynb) | NumPy MSE and gradient, custom gradient descent, three learning rates, comparison with `LinearRegression`, metrics, and error analysis |
| 5 | [README](practice5/README.md) | [Classification](practice5/05_used_cars_classification.ipynb) | Diesel versus other known fuel types, baseline, class weights, undersampling, repeated stratified cross-validation, and confusion matrices |
| 6 | [README](practice6/README.md) | [Text classification](practice6/06_kyrgyz_news_classification.ipynb) | Kyrgyz tokenization and stop words, TF-IDF, `MultinomialNB`, `LogisticRegression`, confusion matrices, and five distinct misclassified articles |
| 7 | [README](practice7/README.md) | [Random forest](practice7/07_used_cars_random_forest.ipynb) | Cross-validated parameter selection, comparison with a decision tree, MDI, permutation importance, and SHAP explanations for three examples with an additivity check |

## Main results

The numbers below come from saved notebook outputs. Models were not retrained during the review on **October 14, 2026**, or the directory reorganization on **October 27, 2026**.

### Practice 1: environment setup and initial analysis

Notebook: [01_used_car_analysis.ipynb](practice1/01_used_car_analysis.ipynb).

Loaded the **4,009 × 12** car dataset and checked column types, missing values, exact duplicates, and fuel categories. Converted prices from text to numbers and documented all 12 original features. Explained the dataset's relevance to predicting car asking prices. Dependency versions are pinned in `requirements.txt`.

### Practice 2: exploratory data analysis

The mean price is **44,553.19 USD** and the median is **31,000 USD**. The distribution has a long right tail.

- Spearman correlation between model year and price: **0.6974**.
- Spearman correlation between mileage and price: **−0.7552**.
- Mann–Whitney test for `accident` groups: **p = 8.31 × 10⁻⁶⁸**, with a rank-based effect size of **−0.3702**.
- Median prices: **35,667.50 USD** without a reported accident and **20,900 USD** with a reported accident.

All three null hypotheses are rejected at the Bonferroni threshold of 0.05 / 3 ≈ 0.01667. Correlation p-values displayed as zero reflect numerical precision. These associations do not establish causation.

### Practice 3: preprocessing

Practices 3 and 4 use six original features: `brand`, `model_year`, `milage`, `fuel_type`, `accident`, and `clean_title`. The train/test split is **3,207/802**, with `random_state=42`.

The custom transformer parses mileage and adds `relative_milage = (milage + 1) / (brand median + 1)`. Medians are learned only from training data; unseen brands use the overall training median. The pipeline produces **72 features**.

Saved outputs show all three tests passing: preservation of fitted statistics and input data; handling of known/unseen brands and missing values; and the complete pipeline on new data.

Increasing training data from 3,207 to 160,350 rows increased median `fit_transform` time from **0.0661 to 1.2838 seconds**, and `transform` time from **0.0307 to 0.6816 seconds**. The custom transformer was the slowest measured transformation step. Repeated rows assess speed, not model quality; timings depend on the computer.

### Practice 4: linear regression with NumPy

| Model | Test MAE, USD | Test RMSE, USD | Test R² |
|---|---:|---:|---:|
| Training-mean baseline | 35,276.16 | 143,144.98 | −0.002 |
| NumPy gradient descent | 24,884.72 | 137,787.54 | 0.071 |
| LinearRegression | 24,705.08 | 137,899.78 | 0.070 |

MAE is the mean absolute price error; RMSE penalizes large errors more strongly. Both are measured in USD, MSE in USD², and R² is dimensionless.

Learning rates of **0.001, 0.01, and 0.1** were tested for 5,000 updates each. **0.1** gave the lowest final training MSE. The custom model's training MSE is **2.03%** higher than scikit-learn's. Full one-hot encoding with an intercept permits non-unique coefficients, so coefficient differences must be considered alongside prediction metrics.

The test set contains prices above the training maximum. The three largest errors account for approximately **94.4%** of the custom regression's sum of squared errors. An extreme price alone does not prove a data error.

### Practice 5: diesel-car classification

The labeled sample contains **3,792 listings**: 116 diesel cars and 3,676 others, an imbalance of **1:31.69**. Unknown fuel types are excluded. Neither `fuel_type` nor `is_diesel` is used as an input feature. Predictors are model year, mileage, brand, transmission, reported accidents, and clean-title status.

Methods use the same 15 training folds: 5 folds × 3 repetitions. Undersampling is applied only within the training portion, and the threshold is fixed in advance at **0.5**. The test set contains 759 listings, including 23 diesel cars.

| Method | Mean CV F1 | Test precision | Test recall | Test F1 |
|---|---:|---:|---:|---:|
| No correction | 0.0517 | 0.0000 | 0.0000 | 0.0000 |
| `class_weight="balanced"` | 0.1794 | 0.0942 | 0.7826 | 0.1682 |
| Random undersampling | 0.1532 | 0.1010 | 0.8696 | 0.1810 |

Class weighting is selected by mean cross-validation F1. Undersampling achieves a higher F1 on the separate test set; the test set is not used to select the method again. High baseline accuracy conceals the failure to identify any diesel cars. Low precision limits automatic recovery of missing fuel labels.

### Practice 6: news topic classification

TF-IDF is fitted only on training data and produces **6,968 features**, using unigrams and bigrams.

| Model | Test accuracy | Test macro-F1 | Test weighted-F1 |
|---|---:|---:|---:|
| DummyClassifier | 0.2500 | 0.0400 | 0.1000 |
| MultinomialNB | 0.5357 | 0.2600 | 0.4619 |
| LogisticRegression | 0.6000 | 0.3619 | 0.5532 |

Logistic regression performs best in this comparison, but F1 is zero for `culture`, `foreign affairs`, `society`, and `sports`. Accuracy alone hides problems with rare classes. Preprocessing preserves the standard Kyrgyz letters U+04E9, U+04AF, and U+04A3, as well as negation, but does not normalize the letter variant U+04CA. This limitation is documented in the notebook conclusion.

### Practice 7: random forest and model interpretation

Notebook: [07_used_cars_random_forest.ipynb](practice7/07_used_cars_random_forest.ipynb).

A decision tree and a random forest predict prices on the same **3,207 training / 802 test** split. Preprocessing is simpler than in practice 4: it omits scaling and `relative_milage`.

| Model | Test MAE, USD | Test RMSE, USD | Test R² |
|---|---:|---:|---:|
| Median baseline | 32,865.97 | 144,218.92 | −0.018 |
| Decision tree | 25,270.65 | 135,584.79 | 0.101 |
| Random forest | 21,808.88 | 135,476.47 | 0.102 |

Three-fold cross-validation selected **300 trees, `max_depth=None`, `min_samples_leaf=1`**, with mean CV MAE of **14,222.80 USD**. The forest reduces test MAE by approximately **13.7%** relative to the tree, but R² remains around 0.10, indicating limited predictive quality.

MDI and permutation importance identify brand, mileage, and model year as the leading features, with different rankings for brand and mileage. SHAP is calculated for three test cars, with waterfall plots and local explanations. One-hot contributions are summed by original feature, and an additivity check confirms that the contributions plus the base value reproduce predictions. Feature importance and SHAP describe model behavior, not causal effects on prices.

## Setup and execution

1. Open the `ml-social-media-sleep` repository directory in VS Code.
2. Install the Python and Jupyter extensions if needed.
3. Run these commands in PowerShell from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `.venv` already exists, skip creating it. Activation is optional because the command explicitly uses the environment's Python executable.

4. Open the desired `.ipynb` from its `practice1/`–`practice7/` directory.
5. Use **Select Kernel** to choose `.venv\Scripts\python.exe`.
6. Confirm that the shared CSV is in `data/raw/` and the two news CSVs are in `practice6/data/`. Loading cells locate these files from the repository root, a practice directory, or the enclosing `ml_course` directory. No absolute machine-specific dataset paths are needed.
7. Run **Restart Kernel → Run All** and save the notebook with its outputs.

The recommended reading order is 1–7. Random-forest parameter tuning in practice 7 may take some time.

Dependencies are pinned in [requirements.txt](requirements.txt): NumPy, pandas, matplotlib, seaborn, SciPy, scikit-learn, ipykernel, and SHAP. Install dependencies from `requirements.txt` before running the notebooks.

## Verification and remaining work

The **October 14, 2026** review covered the seven notebooks' code, Markdown, saved text results, presence of graphical outputs, Python syntax with Jupyter commands accounted for, CSV dimensions, and environment versions. No syntax errors or saved error outputs were found. Plots were not individually inspected visually. A complete rerun and submission to the instructor were not confirmed.

The **October 27, 2026** reorganization moves notebooks into separate practice directories and relocates the news CSVs. Dataset-loading cells and local documentation paths are updated. Saved model outputs are retained; this reorganization does not constitute a complete notebook rerun.

Remaining items from the earlier review:

- **Practice 2:** correct the parentheses in the unparsed-value diagnostic to `(df['price'].notna() & df['price_usd'].isna()).sum()`, and likewise for mileage. Boxplot whiskers do not define a price ceiling. Use the rank-based effect size of −0.370 instead of the p-value in the useful-features table. Replace the mention of U+FFFD with the actual `–` symbol. Add a separate explanation of the heatmap.
- **Practice 3:** fill the conclusion placeholders using the saved test and timing results.
- **Practice 4:** remove or complete old Markdown templates next to the finished conclusions. The metrics cell has a saved output but `execution_count=null`; confirm the result with a complete rerun.
- **Practice 6:** retrain models and update results if the handling of U+04CA is corrected.
- **Practice 7:** `–` remains a separate fuel category; it should not be interpreted as a real fuel type. Changing this rule requires recalculating results.
- Run **Restart & Run All** for every assignment, check the results, and save outputs.

The full assignment guide was unavailable at the previously recorded paths `docs/practical_assignments_1-7.txt` and `docs/assingments.md`, so a complete comparison with the original requirements was not performed. Instructor approval for a separate corpus in practice 6, under the stated single-dataset requirement, remains unconfirmed.
