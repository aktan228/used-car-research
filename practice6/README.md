# Practice 6: Kyrgyz News Topic Classification

**Student:** Aktan Iygilikovich Kashymbekov

**Institution:** International College of IT and Business, Salymbekov University

**Assignment level:** C

## Objective

Prepare a labeled Kyrgyz-language text corpus, build a language-aware preprocessing and TF-IDF workflow, compare two classifiers, and examine actual classification errors.

## Data and files

- Notebook: [06_kyrgyz_news_classification.ipynb](06_kyrgyz_news_classification.ipynb).
- Local CSVs: [qad-train_1000.csv](data/qad-train_1000.csv), with **1,000 rows**, and [qad-test_0500.csv](data/qad-test_0500.csv), with **500 rows**.
- Source: [Kyrgyz Multi-Label Topic Classification](https://github.com/alexeyev/kyrgyz-multi-label-topic-classification), described in the [authors' paper](https://arxiv.org/pdf/2308.15952).
- Content: Kyrgyz-language articles from 24.kg, with columns `id`, `text`, `labels_set`, and `tokenized_text`.

The original corpus permits multiple labels per article. This assignment uses a filtered **single-label** sample. It uses its own split after combining the two source files, so its results are not directly comparable to the original benchmark.

## Workflow

1. Combine the source files and remove empty texts or labels.
2. Retain articles with exactly one topic, remove normalized duplicates and conflicting labels, and keep classes with at least 20 documents.
3. Apply Unicode NFKC normalization, lowercase conversion, URL removal, tokenization, and a small fixed Kyrgyz stop-word list that preserves negation.
4. Create a stratified 80/20 split with `random_state=42`, checking that normalized texts do not overlap.
5. Fit TF-IDF on training texts only, using unigrams and bigrams, `min_df=2`, `max_df=0.95`, and `max_features=15000`.
6. Compare `MultinomialNB` and `LogisticRegression` against a most-frequent-class `DummyClassifier`. Inspect class reports, confusion matrices, and five distinct misclassified articles.

## Recorded results

The following values come from saved notebook outputs.

| Preparation stage | Documents |
|---|---:|
| Original corpus | 1,500 |
| Nonempty texts and labels | 1,477 |
| Single-label articles | 774 |
| After removing conflicts and normalized duplicates | 753 |
| Classes with at least 20 documents | 696 |

The final sample contains **10 classes** and is split into **556 training / 140 test documents**, with no overlapping normalized texts. TF-IDF produces **6,968 features**; neither split contains a document with an all-zero feature vector.

| Model | Test accuracy | Test macro-F1 | Test weighted-F1 |
|---|---:|---:|---:|
| DummyClassifier | 0.2500 | 0.0400 | 0.1000 |
| MultinomialNB | 0.5357 | 0.2600 | 0.4619 |
| LogisticRegression | 0.6000 | 0.3619 | 0.5532 |

Logistic regression performs best in this comparison, with **56** test errors versus **65** for MultinomialNB. The notebook discusses five distinct examples involving medical policy, transport announcements, social assistance, local events, and legal changes.

## Interpretation and limitations

- Logistic regression has zero F1 for `culture`, `foreign affairs`, `society`, and `sports`; overall accuracy hides poor performance on these classes.
- The sample is small and imbalanced. Filtering to one label changes the task from the source corpus's multilabel setting.
- The tokenizer preserves the standard Kyrgyz letters U+04E9, U+04AF, and U+04A3 and negation, but does not normalize the variant U+04CA. Correcting this requires rerunning preprocessing and models before updating results.
- Explanations of individual errors are interpretations of the texts, not verified analyses of model coefficients.

## Run

Open the repository root, `ml-social-media-sleep`, in VS Code. From that root, create the environment if needed and install the pinned dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Skip the first command if `.venv` already exists. Open [06_kyrgyz_news_classification.ipynb](06_kyrgyz_news_classification.ipynb) and select `.venv\Scripts\python.exe` as the Jupyter kernel. Run **Restart Kernel → Run All**, review the outputs, and save the notebook.

The notebook locates its CSV files when run from the repository root, this practice directory, or the enclosing `ml_course` directory. It contains its own required definitions; running another notebook first is unnecessary.

Results above describe saved notebook outputs. This README was written without rerunning the notebook.

[All assignments and shared setup](../README.md) · [Pinned dependencies](../requirements.txt)
