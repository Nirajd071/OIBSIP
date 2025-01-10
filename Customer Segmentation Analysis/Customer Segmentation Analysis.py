# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plot aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})  # Set default font size for readability

# Step 1: Load the dataset
data = pd.read_csv("C:/Users/niraj/Desktop/Internship/Customer Segmentation Analysis/ifood_df.csv")

# Step 2: Data Exploration
print("Dataset Information:")
print(data.info())
print("\nFirst few rows of the dataset:")
print(data.head())

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Step 3: Data Cleaning
# Drop missing values (if any)
data = data.dropna()

# Step 4: Feature Engineering
# Add total spending feature
data['TotalSpending'] = (
    data["MntWines"] + data["MntFruits"] + data["MntMeatProducts"] +
    data["MntFishProducts"] + data["MntSweetProducts"] + data["MntGoldProds"]
)

# Step 5: Descriptive Statistics
print("\nDescriptive Statistics:")
print(data.describe())

# Step 6: Select features for clustering
features = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds",
    "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases",
    "NumStorePurchases", "NumWebVisitsMonth", "TotalSpending"
]

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

# Step 7: Determine optimal number of clusters using Elbow Method
inertia = []
silhouette_scores = []
cluster_range = range(2, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(data_scaled, kmeans.labels_))

# Plot the Elbow curve
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, inertia, marker='o', linestyle='--', label="Inertia")
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.xticks(cluster_range)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plot Silhouette Scores
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='--', color='orange', label="Silhouette Score")
plt.title('Silhouette Scores for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(cluster_range)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Step 8: Apply K-Means clustering
optimal_clusters = 4  # Replace with the number chosen from the elbow or silhouette analysis
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Step 9: Visualize clusters with PCA (2D Projection)
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)
plt.figure(figsize=(12, 8))
sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=data['Cluster'], palette='viridis', s=100)
plt.title('Clusters Visualized with PCA (2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster', loc='upper right')
plt.tight_layout()
plt.show()

# Step 10: Cluster counts
plt.figure(figsize=(10, 6))
sns.countplot(x='Cluster', data=data, palette="viridis")
plt.title('Customer Count per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
for i, count in enumerate(data['Cluster'].value_counts(sort=False)):
    plt.text(i, count + 5, str(count), ha='center', fontsize=12)
plt.tight_layout()
plt.show()

# Step 11: Analyze cluster characteristics
cluster_summary = data.groupby('Cluster').mean()
print("\nCluster Summary:")
print(cluster_summary)

# Step 12: Insights and Recommendations
print("\nInsights:")
for cluster in range(optimal_clusters):
    print(f"Cluster {cluster}:")
    print(cluster_summary.loc[cluster])
    print("--------")

# Step 13: Export the segmented dataset (Optional)
data.to_csv('segmented_customers.csv', index=False)
print("\nSegmented dataset saved as 'segmented_customers.csv'")
