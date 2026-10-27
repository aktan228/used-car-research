# Practice 4: Linear Regression and Gradient Descent with NumPy

**Student:** Aktan Iygilikovich Kashymbekov

**Institution:** International College of IT and Business, Salymbekov University

**Assignment level:** C

## Objective

Implement mean squared error, its gradient, and linear-regression training with NumPy, then compare the resulting model with scikit-learn on identical data.

## Data and files

- Notebook: [04_used_cars_regression.ipynb](04_used_cars_regression.ipynb).
- Shared dataset: [used_cars.csv](../data/raw/used_cars.csv), from Taeef Najib's [Kaggle dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset).
- Target: numeric `price`, the asking price in USD.
- Original features: `brand`, `model_year`, `milage`, `fuel_type`, `accident`, and `clean_title`.
- Split: **3,207 training / 802 test rows**, with `random_state=42`.

## Workflow

1. Parse prices and reproduce the preprocessing from [practice 3](../practice3/README.md), fitting it only on training data.
2. Convert the 72 output features to NumPy arrays and add a column of ones, giving **73 parameters** including the intercept.
3. Implement `mse_loss`, `mse_gradient`, and `gradient_descent`; check them on a small example with a known solution.
4. Compare learning rates **0.001, 0.01, and 0.1**, using 5,000 updates each and plotting convergence.
5. Select the learning rate by final training MSE, compare with `LinearRegression(fit_intercept=False)`, and evaluate a training-mean baseline.
6. Inspect coefficient differences, train/test metrics, prediction plots, and the largest errors.

## Recorded results

The following values come from saved notebook outputs.

The selected learning rate is **0.1**, with final training MSE of **919,563,964.06 USD²**, approximately **2.03%** above scikit-learn's training MSE.

| Model | Test MAE, USD | Test RMSE, USD | Test R² |
|---|---:|---:|---:|
| Training-mean baseline | 35,276.16 | 143,144.98 | −0.002 |
| NumPy gradient descent | 24,884.72 | 137,787.54 | 0.071 |
| LinearRegression | 24,705.08 | 137,899.78 | 0.070 |

Both regressions improve on the baseline, but test R² remains low. Scikit-learn has slightly lower MAE, while the custom model has slightly lower RMSE.

## Interpretation and limitations

- MAE and RMSE are measured in USD; MSE is measured in USD². RMSE is especially sensitive to large errors.
- The test-set maximum price is **2,954,083 USD**, compared with **749,950 USD** in training. The three largest errors account for approximately **94.4%** of the custom model's test sum of squared errors.
- Full one-hot encoding together with an intercept allows non-unique coefficients. Coefficient differences alone do not establish that predictions differ substantially.
- Some older Markdown templates remain alongside completed explanations, and the metrics cell has saved outputs with `execution_count=null`. A fresh full run is needed to confirm execution order and refresh outputs.

## Run

Open the repository root, `ml-social-media-sleep`, in VS Code. From that root, create the environment if needed and install the pinned dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Skip the first command if `.venv` already exists. Open [04_used_cars_regression.ipynb](04_used_cars_regression.ipynb) and select `.venv\Scripts\python.exe` as the Jupyter kernel. Run **Restart Kernel → Run All**, review the outputs, and save the notebook.

The notebook locates its CSV files when run from the repository root, this practice directory, or the enclosing `ml_course` directory. It contains its own required definitions; running another notebook first is unnecessary.

Results above describe saved notebook outputs. This README was written without rerunning the notebook.

[All assignments and shared setup](../README.md) · [Pinned dependencies](../requirements.txt)
