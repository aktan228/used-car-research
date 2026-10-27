# Practice 5: Classification with Imbalanced Classes

**Student:** Aktan Iygilikovich Kashymbekov

**Institution:** International College of IT and Business, Salymbekov University

**Assignment level:** C

## Objective

Identify diesel cars from listing characteristics and compare an uncorrected classifier with two approaches to class imbalance using the same validation scheme.

## Data and files

- Notebook: [05_used_cars_classification.ipynb](05_used_cars_classification.ipynb).
- Shared dataset: [used_cars.csv](../data/raw/used_cars.csv), from Taeef Najib's [Kaggle dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset).
- Target: `is_diesel`, with Diesel = **1** and other known fuel types = **0**. Unknown fuel types are excluded.
- Labeled sample: **3,792 listings**, including **116 diesel** and **3,676 other** cars, an imbalance of **1:31.69**.
- Features: model year, parsed mileage, brand, transmission, reported accidents, and clean-title status. `fuel_type`, `is_diesel`, `model`, and `engine` are excluded from inputs.

## Workflow

1. Prepare the binary labels and visualize their imbalance.
2. Create a stratified 80/20 split with `random_state=42`: **3,033 training / 759 test rows**.
3. Build a preprocessing and `LogisticRegression` pipeline with imputation, scaling, and one-hot encoding.
4. Compare no correction, `class_weight="balanced"`, and random undersampling of the majority class.
5. Use identical **5-fold × 3-repeat stratified cross-validation** splits for every method. Fit preprocessing and apply undersampling only within each training portion.
6. Select the method by mean validation F1 for the diesel class, then evaluate test metrics and confusion matrices at a fixed **0.5** threshold.

## Recorded results

The following values come from saved notebook outputs. Precision, recall, and F1 refer to the positive diesel class. The test set contains **23 diesel cars**.

| Method | Mean CV F1 | Test accuracy | Test precision | Test recall | Test F1 |
|---|---:|---:|---:|---:|---:|
| No correction | 0.0517 | 0.9697 | 0.0000 | 0.0000 | 0.0000 |
| Class weighting | 0.1794 | 0.7655 | 0.0942 | 0.7826 | 0.1682 |
| Random undersampling | 0.1532 | 0.7615 | 0.1010 | 0.8696 | 0.1810 |

**Class weighting** is selected by cross-validation. On the test set it finds **18 of 23** diesel cars, misses **5**, and produces **173 false positives**. Undersampling obtains higher test F1, but test results are not used to repeat model selection.

## Interpretation and limitations

The uncorrected model's high accuracy hides its failure to find any test diesel cars. Both corrections improve recall substantially, with low precision and many false alarms. The models are therefore limited as tools for automatically filling missing fuel labels. Performance on listings with known fuel labels does not establish performance on listings whose fuel type is unknown.

## Run

Open the repository root, `ml-social-media-sleep`, in VS Code. From that root, create the environment if needed and install the pinned dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Skip the first command if `.venv` already exists. Open [05_used_cars_classification.ipynb](05_used_cars_classification.ipynb) and select `.venv\Scripts\python.exe` as the Jupyter kernel. Run **Restart Kernel → Run All**, review the outputs, and save the notebook.

The notebook locates its CSV files when run from the repository root, this practice directory, or the enclosing `ml_course` directory. It contains its own required definitions; running another notebook first is unnecessary.

Results above describe saved notebook outputs. This README was written without rerunning the notebook.

[All assignments and shared setup](../README.md) · [Pinned dependencies](../requirements.txt)
