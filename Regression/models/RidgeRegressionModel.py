from BaseRegressionModel import Regression
from Regularizations.L2 import L2_regularization

class RidgeRegression(Regression):
    def __init__(self, alpha,n_iterations, learn_rate):
        self.regularization=L2_regularization(alpha=alpha)
        super().__init__(n_iterations, learn_rate)
    
    

