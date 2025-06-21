import numpy as np

class L1_regularization():
    def __init__(alpha,self):
        self.alpha=alpha
    
    def __call__(self,w):
        return self.alpha*np.linalg.norm(w,ord=1)
    
    def grad(self,w):
        return self.alpha*np.sign(w)