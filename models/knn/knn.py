import numpy as np 
from collections import Counter

class KNN:
    def __init__(self,k=3,distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
        
    def predict(self,X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self,x):
        distances = [self._calculate_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    def _calculate_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError("Unsupported distance metric")