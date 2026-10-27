# Practice 2: Exploratory Data Analysis of Used Cars

**Student:** Aktan Iygilikovich Kashymbekov

**Institution:** International College of IT and Business, Salymbekov University

**Assignment level:** C

## Objective

Explore car prices and characteristics, identify data-quality issues and potential leakage, and evaluate three statistical hypotheses.

## Data and files

- Notebook: [02_EDA_of_used_cars.ipynb](02_EDA_of_used_cars.ipynb).
- Shared dataset: [used_cars.csv](../data/raw/used_cars.csv), with **4,009 rows and 12 original columns**.
- Source: Taeef Najib's [Used Car Price Prediction Dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset).
- Parsed analysis columns: `price_usd`, `mileage_miles`, and numeric `model_year`.

## Workflow

1. Parse prices and mileage, inspect missing values and duplicates, and examine extreme values.
2. Plot price histograms for the complete dataset and values up to the 99th percentile.
3. Plot price against model year and mileage, and compare accident-report groups with a boxplot.
4. Test two associations with Spearman correlation and compare accident-report groups with the Mann–Whitney U test.
5. Apply a Bonferroni correction for three tests, build a Spearman heatmap, and discuss leakage risks and candidate predictors.

## Recorded results

The following values come from saved notebook outputs.

The mean asking price is **44,553.19 USD**, compared with a median of **31,000 USD**, indicating a long right tail.

| Hypothesis | Method | Association / effect | p-value |
|---|---|---:|---:|
| Spearman correlation between model year and price is zero | Spearman | 0.6974 | Displayed as 0 |
| Spearman correlation between mileage and price is zero | Spearman | −0.7552 | Displayed as 0 |
| Price distributions are the same in the two accident-report groups | Mann–Whitney U | Rank-based effect: −0.3702 | 8.31 × 10⁻⁶⁸ |

All three null hypotheses are rejected at **0.05 / 3 ≈ 0.01667**. A displayed p-value of zero reflects numerical precision, not an exact mathematical zero. The negative rank effect uses the reported-accident group as the first group.

| Accident-report group | Listings | Median price, USD |
|---|---:|---:|
| No accident reported | 2,910 | 35,667.50 |
| At least one accident or damage reported | 986 | 20,900.00 |

Newer model years are associated with higher prices, while higher mileage is associated with lower prices. Reported accidents are associated with lower prices. These are associations rather than causal estimates.

## Interpretation and limitations

- Missing values occur in `fuel_type`, `accident`, and `clean_title`; fuel categories also contain ambiguous values.
- The 99th-percentile histogram hides extreme prices only for visualization; it does not remove those observations from the dataset.
- Boxplot whiskers do not define a maximum valid price, and an extreme observation alone does not prove a data error.
- Learned preprocessing statistics must be fitted on training data in later assignments. Features derived from the target or unavailable at prediction time can cause leakage.
- The notebook's two unparsed-value diagnostic expressions still need their parentheses corrected: `.sum()` should apply to the complete Boolean condition.

## Run

Open the repository root, `ml-social-media-sleep`, in VS Code. From that root, create the environment if needed and install the pinned dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Skip the first command if `.venv` already exists. Open [02_EDA_of_used_cars.ipynb](02_EDA_of_used_cars.ipynb) and select `.venv\Scripts\python.exe` as the Jupyter kernel. Run **Restart Kernel → Run All**, review the outputs, and save the notebook.

The notebook locates its CSV files when run from the repository root, this practice directory, or the enclosing `ml_course` directory. It contains its own required definitions; running another notebook first is unnecessary.

Results above describe saved notebook outputs. This README was written without rerunning the notebook.

[All assignments and shared setup](../README.md) · [Pinned dependencies](../requirements.txt)
