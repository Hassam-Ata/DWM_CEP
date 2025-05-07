import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
import os

# Load the dataset
data = pd.read_csv('customer_data/E-commerce_Customer_Behavior.csv')

# Select relevant features
features = data[['Total Spend', 'Items Purchased']].values

# Compute WCSS (Within-Cluster Sum of Squares) for 1 to 10 clusters
wcss_values = []
cluster_range = range(1, 11)

for k in cluster_range:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(features)
    wcss_values.append(model.inertia_)

# Plot the elbow graph
plt.plot(cluster_range, wcss_values, marker='o')
plt.title('Elbow Method to Determine Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
plt.show()

# Use KneeLocator to find the optimal number of clusters
knee_locator = KneeLocator(cluster_range, wcss_values,
                           curve='convex', direction='decreasing')
optimal_clusters = knee_locator.elbow

# Apply KMeans using the optimal number of clusters
final_model = KMeans(n_clusters=optimal_clusters, random_state=42)
cluster_labels = final_model.fit_predict(features)

# Visualize the resulting clusters
plt.figure(figsize=(8, 6))
cluster_colors = ['blue', 'red', 'green', 'yellow',
                  'purple', 'orange', 'cyan', 'brown', 'pink', 'gray']

for cluster_id in range(optimal_clusters):
    plt.scatter(
        features[cluster_labels == cluster_id, 0],
        features[cluster_labels == cluster_id, 1],
        color=cluster_colors[cluster_id],
        label=f'Cluster {cluster_id + 1}'
    )

# Plot centroids
centroids = final_model.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            color='black', marker='x', s=200, label='Centroids')

# Annotate centroids
for idx, center in enumerate(centroids):
    plt.annotate(f'Centroid {idx + 1}', (center[0], center[1]), textcoords="offset points", xytext=(0, 10),
                 ha='center', color='black', fontweight='bold')

plt.title('K-Means Clustering with Optimal Number of Clusters')
plt.xlabel('Total Spend')
plt.ylabel('Items Purchased')
plt.legend()
plt.grid(True)
plt.show()

# Save the clustered data to a CSV file
# Optional: make cluster labels 1-indexed instead of 0-indexed
data['Cluster'] = cluster_labels + 1

# Ensure output folder exists
output_folder = 'k_means/csv_output'
os.makedirs(output_folder, exist_ok=True)

# Save to CSV without printing
output_path = os.path.join(output_folder, 'customer_clusters.csv')
data[['Customer ID', 'Total Spend', 'Items Purchased', 'Cluster']].to_csv(
    output_path, index=False)


# Summarize clusters
summary = data.groupby('Cluster')[['Total Spend', 'Items Purchased']].agg(
    ['mean', 'min', 'max', 'count'])

# Flatten multi-level column names
summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
summary.reset_index(inplace=True)

# Round 'Items Purchased' stats to integers
summary['Items Purchased_mean'] = summary['Items Purchased_mean'].round().astype(int)
summary['Items Purchased_min'] = summary['Items Purchased_min'].astype(int)
summary['Items Purchased_max'] = summary['Items Purchased_max'].astype(int)

# Save summary to CSV
summary_output_path = os.path.join(output_folder, 'cluster_summary.csv')
summary.to_csv(summary_output_path, index=False)

print("Clustering complete. Summary saved to 'csv_output/cluster_summary.csv'.")
