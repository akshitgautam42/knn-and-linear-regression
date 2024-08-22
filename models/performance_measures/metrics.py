import numpy as np

class PerformanceMetrics:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.classes = np.unique(np.concatenate((y_true, y_pred)))

    def accuracy(self):
        return np.mean(self.y_true == self.y_pred)

    def precision(self, average='macro'):
        precisions = []
        for c in self.classes:
            tp = np.sum((self.y_true == c) & (self.y_pred == c))
            fp = np.sum((self.y_true != c) & (self.y_pred == c))
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        
        if average == 'macro':
            return np.mean(precisions)
        elif average == 'micro':
            tp_total = np.sum(self.y_true == self.y_pred)
            return tp_total / len(self.y_true)

    def recall(self, average='macro'):
        recalls = []
        for c in self.classes:
            tp = np.sum((self.y_true == c) & (self.y_pred == c))
            fn = np.sum((self.y_true == c) & (self.y_pred != c))
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        
        if average == 'macro':
            return np.mean(recalls)
        elif average == 'micro':
            tp_total = np.sum(self.y_true == self.y_pred)
            return tp_total / len(self.y_true)

    def f1_score(self, average='macro'):
        precision = self.precision(average)
        recall = self.recall(average)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0