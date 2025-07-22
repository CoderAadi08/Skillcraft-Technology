# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Sample customer data (agar aapke paas CSV nahi hai)
data = {
    'CustomerID': [101, 102, 103, 104, 105, 106],
    'Annual Income (k$)': [15, 16, 17, 30, 35, 40],
    'Spending Score (1-100)': [39, 81, 6, 77, 40, 76]
}

df = pd.DataFrame(data)

# Data pre-processing
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Add cluster labels to original dataframe
df['Cluster'] = kmeans.labels_

# Visualization
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set1', s=100)
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1–100)')
plt.grid(True)
plt.show()

# Show final data
print(df)