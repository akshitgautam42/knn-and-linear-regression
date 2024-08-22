import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


current_dir = Path(__file__).resolve().parent

# Construct the path to the data file
data_file = current_dir.parent.parent / 'data' / 'external' / 'spotify.csv'

print(data_file)

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
# Additional visualizations

# 1. Bar plot showing the distribution of genres
plt.figure(figsize=(15, 8))
genre_counts = df['track_genre'].value_counts()
sns.barplot(x=genre_counts.index, y=genre_counts.values)
plt.title('Distribution of Genres in the Dataset')
plt.xlabel('Genre')
plt.ylabel('Number of Tracks')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(current_dir / 'figures' / 'genre_distribution.png')
plt.close()

# 2. Analysis of the 'explicit' feature in relation to genres
plt.figure(figsize=(15, 8))
explicit_by_genre = df.groupby('track_genre')['explicit'].mean().sort_values(ascending=False)
sns.barplot(x=explicit_by_genre.index, y=explicit_by_genre.values)
plt.title('Proportion of Explicit Tracks by Genre')
plt.xlabel('Genre')
plt.ylabel('Proportion of Explicit Tracks')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(current_dir / 'figures' / 'explicit_by_genre.png')
plt.close()

# 3. Heatmap of average feature values by genre
features_to_plot = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
                    'instrumentalness', 'liveness', 'valence', 'tempo']
genre_features = df.groupby('track_genre')[features_to_plot].mean()
plt.figure(figsize=(15, 12))
sns.heatmap(genre_features, cmap='YlGnBu', annot=False)
plt.title('Average Feature Values by Genre')
plt.tight_layout()
plt.savefig(current_dir / 'figures' / 'genre_features_heatmap.png')
plt.close()

print("Additional visualizations have been saved in the 'figures' directory.")