WCHS: Weighted Contribution-based Hybrid Sampling
WCHS is a Python implementation of the Weighted Contribution-based Hybrid Sampling method for imbalanced classification in microbial data. The method introduces a dual-stage contribution assessment framework that jointly quantifies the information value and noise level of each sample, with distinct criteria for real samples in the sampling stage and synthetic candidates in the filtering stage.

Features
KPCA-based dimensionality reduction: Reduces high-dimensional microbial data while preserving nonlinear structures
Dual-stage contribution assessment:
Sampling stage: Evaluates real samples using weighted entropy (WInf) and reverse nearest neighbor entropy with local density (DNoi)
Filtering stage: Re-evaluates synthetic candidates using class purity (WInf) and proximity to the majority class (DNoi)
Hybrid sampling:
Undersampling: Retains high-contribution majority samples
Oversampling: Generates synthetic candidates from high-contribution minority samples
Filtering: Retains top-scores synthetic samples

Installation
Requirements
```text
Python == 3.10
numpy == 1.24.4
scikit-learn == 1.3.2
pandas == 2.0.3
matplotlib == 3.7.5
```

git clone https://github.com/SmaleMT/WCHS.git
cd WCHS
pip install -r requirements.txt

Quick Start
from sklearn.datasets import make_classification
from WCHS import WCHS
from data-process import evaluate_dataset

# Generate imbalanced dataset
X, y = make_classification(
    n_samples=300,
    n_features=60,
    weights=[0.9, 0.1],  # 90% negative, 10% positive
    random_state=42
)
evaluate_dataset(X,y)

Core Components
1. KPCA Dimensionality Reduction
KPCA is applied before sampling to reduce dimensionality while capturing nonlinear structures.
2. WCHS Hybrid Sampling
WCHS is applied to balance dataset.

Directory Structure
WCHS/
├── _Datasets.zip    # datasets
├── data-process.py  # Data process: Split, Normalize, KPCA, evaluate
├── WCHS.py          # Undersampling, oversampling, filtering
└── README.md

Citation

