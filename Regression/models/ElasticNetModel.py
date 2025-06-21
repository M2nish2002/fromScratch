from .BaseRegressionModel import Regression
from ..Regularizations.L2L1 import L2_L1_regularization

class ElasticNet(Regression):
    def __init__(self, alpha,L1_ratio,n_iterations, learn_rate):
        self.regularization=L2_L1_regularization(alpha=alpha,L1_ratio=L1_ratio)
        super().__init__(n_iterations, learn_rate)