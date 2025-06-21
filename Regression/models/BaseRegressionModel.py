import math
import numpy as np
class Regression():
    def __init__(self,n_iterations,learn_rate):
          self.n_iterations=n_iterations
          self.learn_rate=learn_rate
    
    def weight_initia(self,n_features):
         limit=1/math.sqrt(n_features)
         self.w=np.random.uniform(-limit,limit,(n_features,))
    
    def fit(self,X,y):
         X=np.insert(X,0,1,axis=1)
         self.training_error=[]
         self.weight_initia(n_features=X.shape[1])
         for i in range(1,self.n_iterations):
              y_pred=X.dot(self.w)
              mse=np.mean(0.5*(y-y_pred)**2)+self.regularization(self.w)
              self.training_error.append(mse)
              grad_w=-(y-y_pred).dot(X)+self.regularization.grad(self.w)
              self.w-=self.learn_rate*grad_w
    
    def prediction(self,X):
         X=np.insert(X,0,1,axis=1)
         y_pred=X.dot(self.w)
         return y_pred
         
         


     
     
        
