import numpy as np

class L2_L1_regularization():
    def __init__(self,alpha,L1_ratio):
        self.alpha=alpha
        self.L1_ratio=L1_ratio
    def __call__(self,w):
        L1_penalty=self.L1_ratio*self.alpha*np.linalg.norm(w,ord=1)
        L2_penalty=(1-self.L1_ratio)*self.alpha*0.5*w.T.dot(w)
        return L1_penalty+L2_penalty
    def grad(self,w):
        return self.alpha*(self.L1_ratio*np.sign(w)+(1-self.L1_ratio)*w)

