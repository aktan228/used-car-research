# Practice 3: Custom Transformer and Preprocessing Pipeline

**Student:** Aktan Iygilikovich Kashymbekov

**Institution:** International College of IT and Business, Salymbekov University

**Assignment level:** C

## Objective

Implement a scikit-learn transformer for a domain-specific feature, combine it with standard preprocessing in a `Pipeline`, test its behavior, and measure performance as the input grows.

## Data and files

- Notebook: [03_used_cars_pipeline.ipynb](03_used_cars_pipeline.ipynb).
- Shared dataset: [used_cars.csv](../data/raw/used_cars.csv), from Taeef Najib's [Kaggle dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset).
- Input features: `brand`, `model_year`, `milage`, `fuel_type`, `accident`, and `clean_title`.
- Split: **3,207 training / 802 test rows**, with `test_size=0.2` and `random_state=42`.

## Workflow

1. Split the data before learning preprocessing statistics.
2. Implement `BrandMileageTransformer` using `BaseEstimator` and `TransformerMixin`.
3. Parse mileage and learn brand-specific mileage medians on training data. Use the overall training median for unseen brands.
4. Add `relative_milage = (milage + 1) / (brand median + 1)`.
5. Apply median imputation and `StandardScaler` to numeric columns; apply constant imputation and one-hot encoding to categorical columns.
6. Test the transformer and complete pipeline, then benchmark fitting and transformation on repeated training data.

## Recorded results

The following values come from saved notebook outputs.

The pipeline produces **72 features** for both training and test data. All three checks have passing outputs:

- Transformation preserves fitted statistics and does not mutate the input.
- Known brands, unseen brands, and missing mileage are handled as expected.
- The complete pipeline handles new categories and missing values while preserving fitted imputer and encoder statistics.

| Input rows | Median fit_transform, s | Median transform, s |
|---|---:|---:|
| 3,207 | 0.0661 | 0.0307 |
| 160,350 | 1.2838 | 0.6816 |

The custom transformer is the slowest measured transformation stage. Timings are medians of three measurements and depend on the computer and environment. Repeating rows measures computational performance, not predictive quality or generalization.

This preprocessing is reused conceptually in [practice 4](../practice4/README.md); practice 4 contains its own definitions and does not require running this notebook first. The final conclusion in this notebook still contains template fields to complete using the saved results.

## Run

Open the repository root, `ml-social-media-sleep`, in VS Code. From that root, create the environment if needed and install the pinned dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Skip the first command if `.venv` already exists. Open [03_used_cars_pipeline.ipynb](03_used_cars_pipeline.ipynb) and select `.venv\Scripts\python.exe` as the Jupyter kernel. Run **Restart Kernel → Run All**, review the outputs, and save the notebook.

The notebook locates its CSV files when run from the repository root, this practice directory, or the enclosing `ml_course` directory. It contains its own required definitions; running another notebook first is unnecessary.

Results above describe saved notebook outputs. This README was written without rerunning the notebook.

[All assignments and shared setup](../README.md) · [Pinned dependencies](../requirements.txt)
