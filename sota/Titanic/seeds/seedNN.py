from sklearn.neural_network import MLPClassifier
import numpy as np

# --OPTION--
class Model:
    def __init__(self):
      self.classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42) 
    
    def fit(self, X, y):
      self.classifier.fit(X, y)

    def predict(self, X):
       return self.classifier.predict(X)



