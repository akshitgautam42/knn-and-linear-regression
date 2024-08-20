import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


current_dir = Path(__file__).resolve().parent

# Construct the path to the data file
data_file = current_dir.parent.parent / 'data' / 'external' / 'spotify.csv'

# Load the data
df = pd.read_csv(data_file)

# Display basic information about the dataset
print(df.info())
print("\nSample of the data:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Display summary statistics
print("\nSummary statistics:")
print(df.describe())

# Set up the plotting style
plt.style.use('ggplot')


# Create a function to plot histograms for numerical features
def plot_histograms(df, features, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        sns.histplot(data=df, x=feature, ax=axes[i], kde=True)
        axes[i].set_title(f'Distribution of {feature}')
    
    plt.tight_layout()
    plt.savefig(current_dir / 'figures' / 'feature_distributions.png')
    plt.close()

# Select numerical features for visualization
numerical_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
                      'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']

# Plot histograms
plot_histograms(df, numerical_features, 3, 4)

# Create a correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.savefig(current_dir / 'figures' / 'correlation_heatmap.png')
plt.close()

# Create box plots for each numerical feature grouped by genre
plt.figure(figsize=(15, 10))
df.boxplot(column=numerical_features, by='track_genre', figsize=(15, 10))
plt.title('Box Plots of Numerical Features by Genre')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(current_dir / 'figures' / 'boxplots_by_genre.png')
plt.close()

# Create a pair plot for a subset of features
subset_features = ['danceability', 'energy', 'loudness', 'tempo', 'track_genre']
sns.pairplot(df[subset_features], hue='track_genre')
plt.savefig(current_dir / 'figures' / 'pairplot.png')
plt.close()