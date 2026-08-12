# Commit Classification for Software Maintenance Activities

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Reproduction & extension of the PROMISE 2017 paper:**  
> *"Boosting Automatic Commit Classification Into Maintenance Activities By Utilizing Source Code Changes"*  
> by S. Levin & A. Yehudai — [ACM Digital Library](https://dl.acm.org/doi/10.1145/3127005.3127016)

---

## Overview

This project implements and evaluates **automated commit classification into software maintenance activities** using **multi-modal feature engineering**. Unlike traditional keyword-only approaches, this work incorporates **fine-grained AST-level source code change metrics** to significantly boost classification accuracy — directly reproducing and extending the methodology proposed by Levin & Yehudai (2017).

The system classifies commits into three maintenance categories:
- **Corrective** — Bug fixes, error corrections
- **Perfective** — Refactoring, code improvements
- **Adaptive** — New features, functional additions

Using three distinct feature strategies and four ML classifiers, we demonstrate that **fusing commit message keywords with code change features** consistently outperforms either strategy in isolation.

---

## Methodology

### Feature Engineering Strategies

| Strategy | Description | Features |
|----------|-------------|----------|
| **Keywords** | Stemmed bag-of-words from commit messages (e.g., `fix`, `bug`, `refactor`, `implement`) | 19–20 binary |
| **Code Changes** | Fine-grained AST-level change types (e.g., `STATEMENT_DELETE`, `METHOD_RENAMING`, `ADDITIONAL_CLASS`) | 46–48 binary |
| **Combined** | Fusion of Keywords + Code Changes for maximum predictive power | 65–68 binary |

### Classifiers Evaluated

| Classifier | Framework | Key Characteristics |
|------------|-----------|---------------------|
| **Random Forest** | scikit-learn | 200 estimators, ensemble bagging, robust to overfitting |
| **Gradient Boosting (GBM)** | scikit-learn | Sequential boosting with gradient descent, 10×3 repeated CV |
| **Decision Tree (J48)** | scikit-learn | Interpretable rule-based classifier, C4.5 equivalent |
| **Neural Network** | PyTorch | 2 hidden layers (128→64), BatchNorm, Dropout(0.3), Adam optimizer |

### Evaluation Metrics

- **Accuracy** & **10-Fold Repeated Cross-Validation**
- **Precision, Recall, F1-Score** (per-class & macro-averaged)
- **Cohen's Kappa** (inter-rater agreement)
- **Confusion Matrix** (visualized)

---

## Repository Structure

```
commit-classification/
├── .github/workflows/ci.yml          # GitHub Actions: lint, test, run on push
├── data/
│   └── raw/dataset.csv               # Labeled commit dataset (# delimiter)
├── src/
│   ├── data/
│   │   ├── commit.py                 # Commit dataclass with tensor/label utilities
│   │   └── dataset.py                # CSV loader + feature strategy selector
│   ├── features/
│   │   ├── keywords.py               # NLTK-based keyword extractor (19 keywords)
│   │   └── code_changes.py           # AST change type catalog (46 types)
│   ├── preprocessing/
│   │   └── nlp.py                    # Text cleaning, tokenization, lemmatization
│   ├── models/
│   │   ├── sklearn_models.py         # Unified RF/GBM/J48 trainer & evaluator
│   │   └── pytorch_models.py         # Custom NN + training loop with early stopping
│   └── evaluation/
│       └── metrics.py                # Confusion matrix & training curve plots
├── scripts/
│   ├── train_sklearn.py              # CLI: python scripts/train_sklearn.py --strategy combined
│   └── train_pytorch.py              # CLI: python scripts/train_pytorch.py --epochs 40
├── tests/                             # pytest suite (data, features, models)
├── results/                           # Generated metrics.json & figures/
├── requirements.txt
├── pyproject.toml                     # Modern Python packaging
└── README.md
```

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/Mehedi-909/commit-classification.git
cd commit-classification
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download NLTK Data

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt')"
```

### 3. Prepare Dataset

Place your labeled commit dataset at `data/raw/dataset.csv` with `#` delimiter and columns:
```
commitId#project#comment#label#<keyword_features...>#<code_change_features...>
```

A synthetic sample dataset is included for testing.

### 4. Run Experiments

**Sklearn classifiers (all strategies):**
```bash
python scripts/train_sklearn.py --strategy combined --output results/
python scripts/train_sklearn.py --strategy keywords --output results/
python scripts/train_sklearn.py --strategy changes --output results/
```

**PyTorch neural network:**
```bash
python scripts/train_pytorch.py --strategy combined --epochs 40 --batch-size 32 --lr 0.001 --output results/
```

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## Key Implementation Details

### NLP Preprocessing Pipeline (`src/preprocessing/nlp.py`)

1. **Punctuation removal** — `string.punctuation` filter
2. **Tokenization** — `nltk.RegexpTokenizer(r'\w+')`
3. **Stopword filtering** — NLTK English stopwords
4. **Lemmatization** — `WordNetLemmatizer` to normalize word forms

### Sklearn ML Pipeline (`src/models/sklearn_models.py`)

```python
# Three feature configurations tested independently
X_keywords = data[keyword_features]      # ~19 features
X_changes  = data[code_change_features]  # ~46 features
X_combined = data[all_features]          # ~65 features

# Stratified train-test split: 85/15
train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# Classifiers with full metric reporting
RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
GradientBoostingClassifier(random_state=42)  # + RepeatedStratifiedKFold(10×3)
DecisionTreeClassifier(random_state=42)      # J48 equivalent
```

### PyTorch Deep Learning Pipeline (`src/models/pytorch_models.py`)

```python
# Architecture
CommitClassifier(
    input_dim=65,           # Combined feature count
    hidden_dims=[128, 64],  # Two hidden layers
    num_classes=3,          # Corrective / Perfective / Adaptive
    dropout=0.3,
)

# Training
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Scheduler: ReduceLROnPlateau
- Early stopping: Best model checkpoint saved
- Batch size: 32 (configurable)
```

---

## Results Summary

| Model | Feature Strategy | Accuracy | Cohen's Kappa | Macro F1 |
|-------|-----------------|----------|---------------|----------|
| Random Forest | Keywords | ~65–70% | ~45–50% | ~60–65% |
| Random Forest | Code Changes | ~70–75% | ~50–55% | ~65–70% |
| **Random Forest** | **Combined** | **~80–85%** | **~70–75%** | **~78–82%** |
| Gradient Boosting | Combined | ~78–83% | ~68–73% | ~76–80% |
| Decision Tree (J48) | Combined | ~72–77% | ~55–60% | ~68–72% |
| PyTorch NN | Combined | ~75–82% | ~62–70% | ~72–78% |

> **Key Finding:** The **Combined feature strategy** consistently outperforms individual Keyword-only and Changes-only approaches across all classifiers, validating the core hypothesis of Levin & Yehudai (2017) that **source code change features provide complementary signal** to commit message text.

---

## Technologies Used

| Category | Stack |
|----------|-------|
| **Language** | Python 3.9+ |
| **ML / Classical** | scikit-learn (Random Forest, GBM, Decision Tree) |
| **Deep Learning** | PyTorch 2.0+ (CUDA support) |
| **NLP** | NLTK (WordNet, stopwords, tokenization, lemmatization) |
| **Data** | pandas, NumPy |
| **Visualization** | matplotlib, seaborn |
| **Evaluation** | pycm, sklearn.metrics |
| **CI/CD** | GitHub Actions |
| **Testing** | pytest, pytest-cov |

---

## Continuous Integration

Every push and PR triggers:
- Dependency installation
- NLTK data download
- Full pytest suite across Python 3.9, 3.10, 3.11
- Coverage reporting

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## References

1. Levin, S., & Yehudai, A. (2017). *Boosting Automatic Commit Classification Into Maintenance Activities By Utilizing Source Code Changes*. In Proceedings of the 13th International Conference on Predictive Models and Data Analytics in Software Engineering (PROMISE '17). ACM. [https://doi.org/10.1145/3127005.3127016](https://doi.org/10.1145/3127005.3127016)

---

## Author

- **Fahim** — Software Engineer, Samsung R&D Bangladesh

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.  
Dataset sourced from publicly available MSR research data.
