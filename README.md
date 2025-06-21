# fromScratch

## 🚀 Models Implemented

| Model | Description | File |
|-------|-------------|------|
| **Linear Regression** | No regularization, just MSE minimization | [BaseRegressionModel.py](Regression/models/BaseRegressionModel.py) |
| **Ridge Regression** | Adds L2 regularization (shrinks weights) | [RidgeRegressionModel.py](Regression/models/RidgeRegressionModel.py) |
| **Lasso Regression** | Adds L1 regularization (sparse weights) | [LassoRegressionModel.py](Regression/models/LassoRegressionModel.py) |
| **Elastic Net** | Combination of L1 and L2 | [ElasticNetModel.py](Regression/models/ElasticNetModel.py) |

---

## ⚙️ Regularization Classes

| Regularization | Description | File |
|----------------|-------------|------|
| **L1 (Lasso)** | Encourages sparsity using L1 norm | [L1.py](Regression/Regularizations/L1.py) |
| **L2 (Ridge)** | Shrinks weights using L2 norm | [L2.py](Regression/Regularizations/L2.py) |
| **Elastic Net** | Combination of L1 and L2 penalties | [L2L1.py](Regression/Regularizations/L2L1.py) |

---
