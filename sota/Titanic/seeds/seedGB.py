from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

# --OPTION--
class Model:
    def __init__(self):
      self.classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42) 
    
    def fit(self, X, y):
      self.classifier.fit(X, y)

    def predict(self, X):
       return self.classifier.predict(X)



