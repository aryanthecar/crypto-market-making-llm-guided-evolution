from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --OPTION--
class Model:
    def __init__(self):
      self.classifier = RandomForestClassifier(n_estimators=100, random_state=42) 
    
    def fit(self, X, y):
      self.classifier.fit(X, y)

    def predict(self, X):
       return self.classifier.predict(X)



