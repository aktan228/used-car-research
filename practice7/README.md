# Practice 7: Random Forest and Model Interpretation

**Student:** Aktan Iygilikovich Kashymbekov

**Institution:** International College of IT and Business, Salymbekov University

**Assignment level:** C

## Objective

Compare a decision tree and a random forest on the same regression task, tune forest parameters with cross-validation, and explain predictions using feature importance and SHAP.

## Data and files

- Notebook: [07_used_cars_random_forest.ipynb](07_used_cars_random_forest.ipynb).
- Shared dataset: [used_cars.csv](../data/raw/used_cars.csv), from Taeef Najib's [Kaggle dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset).
- Target: asking price in USD.
- Features: `brand`, `model_year`, `milage`, `fuel_type`, `accident`, and `clean_title`.
- Split: **3,207 training / 802 test rows**, with `test_size=0.2` and `random_state=42`.

## Workflow

1. Parse prices and mileage, handle invalid numeric values, and prepare categorical inputs.
2. Build a preprocessing pipeline with median numeric imputation, constant categorical imputation, and dense one-hot encoding. Unlike practice 4, this pipeline omits scaling and relative mileage.
3. Search **8 parameter combinations** with shuffled three-fold cross-validation on training data: 100/300 trees, maximum depth 10/None, and minimum leaf size 1/5. Select by MAE.
4. Compare the selected forest with a decision tree using the same selected depth and leaf constraints, plus a training-median baseline.
5. Aggregate impurity-based importance (MDI) by original feature and compare it with test-set permutation importance over 10 repetitions.
6. Explain three test cars using SHAP waterfall plots, aggregate one-hot contributions by original feature, and verify additivity.

## Recorded results

The following values come from saved notebook outputs.

Cross-validation selects **`n_estimators=300`, `max_depth=None`, `min_samples_leaf=1`**, with mean CV MAE of **14,222.80 USD**.

| Model | Test MAE, USD | Test RMSE, USD | Test R² |
|---|---:|---:|---:|
| Training-median baseline | 32,865.97 | 144,218.92 | −0.018 |
| Decision tree | 25,270.65 | 135,584.79 | 0.101 |
| Random forest | 21,808.88 | 135,476.47 | 0.102 |

The forest reduces test MAE by **3,461.76 USD**, approximately **13.7%**, relative to the tree. Test R² remains near 0.10, indicating limited predictive quality.

| Feature | MDI | Permutation increase in MAE, USD |
|---|---:|---:|
| `brand` | 0.455 | 7,760.00 |
| `milage` | 0.432 | 12,353.39 |
| `model_year` | 0.080 | 4,827.46 |

The methods agree on the three leading features but reverse the order of brand and mileage. SHAP is calculated for **three cars**, producing a **3 × 70** contribution matrix before aggregation. The saved additivity checks pass.

## Interpretation and limitations

- MDI and permutation importance use different scales and measure different aspects of model behavior; correlated features may share importance.
- SHAP contributions and feature importances explain model predictions, not causal effects or the true price of a car.
- The fuel value `–` remains a separate category in the current pipeline. It should not be interpreted as a real fuel type; changing its treatment requires recalculating results.
- Parameter tuning and permutation importance may take noticeable time when rerunning the notebook.

## Run

Open the repository root, `ml-social-media-sleep`, in VS Code. From that root, create the environment if needed and install the pinned dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Skip the first command if `.venv` already exists. Open [07_used_cars_random_forest.ipynb](07_used_cars_random_forest.ipynb) and select `.venv\Scripts\python.exe` as the Jupyter kernel. Run **Restart Kernel → Run All**, review the outputs, and save the notebook.

The notebook locates its CSV files when run from the repository root, this practice directory, or the enclosing `ml_course` directory. It contains its own required definitions; running another notebook first is unnecessary.

Results above describe saved notebook outputs. This README was written without rerunning the notebook.

[All assignments and shared setup](../README.md) · [Pinned dependencies](../requirements.txt)
