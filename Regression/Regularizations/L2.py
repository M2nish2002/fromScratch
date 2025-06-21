class L2_regularization():
    def __init__(self,alpha):
        self.alpha=alpha
    def __call__(self,w):
        return self.alpha*0.5*w.T.dot(w)
    def grad(self,w):
        return self.alpha*w