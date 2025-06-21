from BaseRegressionModel import Regression
from Regularizations.L1 import L1_regularization

class LassoRegression(Regression):
    def __init__(self, alpha,n_iterations, learn_rate):
        self.regularization=L1_regularization(alpha=alpha)
        super().__init__(n_iterations, learn_rate)