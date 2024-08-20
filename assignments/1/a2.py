import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Import our KNN class
from models.knn.knn import KNN

# Load the data
current_dir = Path(__file__).resolve().parent
data_file = current_dir.parent.parent / 'data' / 'external' / 'spotify.csv'
df = pd.read_csv(data_file)

# Select features and target
features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
            'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
X = df[features]
y = df['track_genre']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the KNN model
knn = KNN(k=5, distance_metric='euclidean')
knn.fit(X_train_scaled, y_train)

# Evaluate the model
metrics = knn.evaluate(X_test_scaled, y_test)

print("KNN Performance Metrics:")
for metric, value in metrics.items():
    print(f"{metric.capitalize()}: {value:.4f}")

# Optional: Test with different k values
k_values = [1, 3, 5, 7, 9, 11]
for k in k_values:
    knn = KNN(k=k, distance_metric='euclidean')
    knn.fit(X_train_scaled, y_train)
    metrics = knn.evaluate(X_test_scaled, y_test)
    print(f"\nKNN (k={k}) Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")