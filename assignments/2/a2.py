import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Now we can import from models
from models.knn.knn import KNN
from models.performance_measures.metrics import PerformanceMetrics 

def correlation_feature_selection(X, y, n_features=5):
    correlations = []
    for feature in X.columns:
        corr = np.abs(np.corrcoef(X[feature], y)[0, 1])
        correlations.append((feature, corr))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    selected_features = [feat for feat, _ in correlations[:n_features]]
    return selected_features

def load_and_preprocess_data(n_features=5):
    current_dir = Path(__file__).resolve().parent
    data_file = current_dir.parent.parent / 'data' / 'external' / 'spotify.csv'
    df = pd.read_csv(data_file)
    
    features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
                'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
    X = df[features]
    y = df['track_genre']
    
    # Select the most correlated features
    selected_features = correlation_feature_selection(X, pd.factorize(y)[0], n_features)
    X_selected = X[selected_features]
    
    print(f"Selected features: {selected_features}")
    
    # Normalize features
    X_selected = (X_selected - X_selected.mean()) / X_selected.std()
    
    return X_selected.values, y.values, selected_features

def evaluate_knn(X, y, k, distance_metric, train_size=0.8):
    # Split the data
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(len(X) * train_size)
    X_train, X_val = X[indices[:train_size]], X[indices[train_size:]]
    y_train, y_val = y[indices[:train_size]], y[indices[train_size:]]

    # Train and predict
    knn = KNN(k=k, distance_metric=distance_metric)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_val)

    # Evaluate
    metrics = PerformanceMetrics(y_val, predictions)
    return {
        'accuracy': metrics.accuracy(),
        'f1_macro': metrics.f1_score('macro'),
        'precision_macro': metrics.precision('macro'),
        'recall_macro': metrics.recall('macro'),
        'f1_micro': metrics.f1_score('micro'),
        'precision_micro': metrics.precision('micro'),
        'recall_micro': metrics.recall('micro')
    }

def main():
    X, y, selected_features = load_and_preprocess_data(n_features=5)
    
    k_values = [ 5, 7, 9]
    distance_metrics = ['euclidean', 'manhattan']

    for k in k_values:
        for metric in distance_metrics:
            print(f"\nEvaluating KNN with k={k} and {metric} distance:")
            results = evaluate_knn(X, y, k, metric)
            for metric_name, value in results.items():
                print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()