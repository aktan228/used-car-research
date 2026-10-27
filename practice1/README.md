# Practice 1: Environment Setup and Initial Data Analysis

**Student:** Aktan Iygilikovich Kashymbekov

**Institution:** International College of IT and Business, Salymbekov University

**Assignment level:** C

## Objective

Set up a reproducible Python environment, select a dataset for the semester project, inspect its structure and quality, and document every original feature.

## Data and files

- Notebook: [01_used_car_analysis.ipynb](01_used_car_analysis.ipynb).
- Shared dataset: [used_cars.csv](../data/raw/used_cars.csv), containing **4,009 listings and 12 columns**.
- Source: **Used Car Price Prediction Dataset**, published by Taeef Najib on [Kaggle](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset).
- Project target: the advertised asking price, `price`, in USD.

Each row is a car listing. The original spelling `milage` is retained. This CSV is also used in practices 2–5 and 7.

## Workflow

1. Load the CSV with pandas and inspect its shape and column types.
2. Count missing values and exact duplicate rows.
3. Inspect fuel categories and descriptive statistics.
4. Remove dollar signs and separators from `price`, then convert it to numeric values.
5. Describe the features and explain why the data supports a used-car price prediction project.

## Data dictionary

| Column | Description | Semantic type |
|---|---|---|
| `brand` | Manufacturer | Categorical |
| `model` | Model and trim | Categorical |
| `model_year` | Model year | Numeric |
| `milage` | Mileage in miles | Numeric, stored as text |
| `fuel_type` | Fuel type | Categorical |
| `engine` | Engine description | Text |
| `transmission` | Transmission description | Categorical |
| `ext_col` | Exterior color | Categorical |
| `int_col` | Interior color | Categorical |
| `accident` | Reported accidents or damage | Categorical |
| `clean_title` | Clean-title status in the listing | Categorical |
| `price` | Asking price in USD | Numeric, stored as text |

## Recorded results

The following values come from saved notebook outputs.

| Check | Result |
|---|---:|
| Rows / columns | 4,009 / 12 |
| Exact duplicate rows | 0 |
| Missing `fuel_type` | 170 |
| Missing `accident` | 113 |
| Missing `clean_title` | 596 |
| Mean asking price | 44,553.19 USD |
| Median asking price | 31,000 USD |
| Minimum / maximum asking price | 2,000 / 2,954,083 USD |

The dataset combines numeric, categorical, and text features with missing values and inconsistent formatting, making it suitable for subsequent preprocessing and modeling assignments. A missing accident report or title status must not be interpreted as a confirmed negative answer. Asking prices may differ from final sale prices.

## Run

Open the repository root, `ml-social-media-sleep`, in VS Code. From that root, create the environment if needed and install the pinned dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Skip the first command if `.venv` already exists. Open [01_used_car_analysis.ipynb](01_used_car_analysis.ipynb) and select `.venv\Scripts\python.exe` as the Jupyter kernel. Run **Restart Kernel → Run All**, review the outputs, and save the notebook.

The notebook locates its CSV files when run from the repository root, this practice directory, or the enclosing `ml_course` directory. It contains its own required definitions; running another notebook first is unnecessary.

Results above describe saved notebook outputs. This README was written without rerunning the notebook.

[All assignments and shared setup](../README.md) · [Pinned dependencies](../requirements.txt)
