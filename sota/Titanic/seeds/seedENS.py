from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
import numpy as np

# --OPTION--
class Model:
    def __init__(self):
      # Create individual classifiers with different algorithms
      lr = LogisticRegression(max_iter=1000, random_state=42)
      svm = SVC(probability=True, random_state=42)
      knn = KNeighborsClassifier(n_neighbors=5)
      ada = AdaBoostClassifier(n_estimators=50, random_state=42)
      
      # Combine them into a voting ensemble
      self.classifier = VotingClassifier(
          estimators=[('lr', lr), ('svm', svm), ('knn', knn), ('ada', ada)],
          voting='hard'
      )
    
    def fit(self, X, y):
      self.classifier.fit(X, y)

    def predict(self, X):
       return self.classifier.predict(X)
