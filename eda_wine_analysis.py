"""
Week 2: Exploratory Data Analysis and Visualization
Dataset: UCI Wine Recognition dataset (available through scikit-learn).
Run: python eda_wine_analysis.py
Requirements: pandas, numpy, matplotlib, scikit-learn
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

os.makedirs("outputs", exist_ok=True)

# 1. Load and inspect data
wine = load_wine(as_frame=True)
df = wine.frame.copy()
df.rename(columns={"target": "wine_class"}, inplace=True)
features = list(wine.feature_names)
df["wine_class_name"] = df["wine_class"].map(
    {i: name for i, name in enumerate(wine.target_names)}
)

print("Shape:", df.shape)
print("\\nMissing values:\\n", df.isnull().sum())
print("\\nSummary statistics:\\n", df[features].describe())
print("\\nClass counts:\\n", df["wine_class_name"].value_counts())

# 2. Class distribution
df["wine_class_name"].value_counts().reindex(wine.target_names).plot(kind="bar")
plt.title("Number of Samples in Each Wine Class")
plt.xlabel("Wine class"); plt.ylabel("Number of samples")
plt.tight_layout(); plt.savefig("outputs/class_distribution.png", dpi=180); plt.close()

# 3. Feature distributions
for feature in ["alcohol", "malic_acid", "color_intensity"]:
    plt.figure(figsize=(7, 4))
    for name in wine.target_names:
        values = df.loc[df.wine_class_name == name, feature]
        plt.hist(values, bins=12, alpha=0.5, label=name)
    plt.title(f"{feature.replace('_', ' ').title()} by Wine Class")
    plt.xlabel(feature); plt.ylabel("Frequency"); plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/{feature}_distribution.png", dpi=180)
    plt.close()

# 4. Correlation matrix
corr = df[features].corr()
plt.figure(figsize=(9, 7))
image = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(image, label="Correlation")
plt.xticks(range(len(features)), features, rotation=90, fontsize=7)
plt.yticks(range(len(features)), features, fontsize=7)
plt.title("Correlation Heatmap")
plt.tight_layout(); plt.savefig("outputs/correlation_heatmap.png", dpi=180); plt.close()

# 5. Scatter plot
for name in wine.target_names:
    part = df[df.wine_class_name == name]
    plt.scatter(part["flavanoids"], part["color_intensity"], label=name, alpha=0.75)
plt.title("Flavanoids vs Color Intensity")
plt.xlabel("Flavanoids"); plt.ylabel("Color intensity"); plt.legend()
plt.tight_layout(); plt.savefig("outputs/feature_scatter.png", dpi=180); plt.close()

# 6. Standardization, PCA and exploratory clustering
X_scaled = StandardScaler().fit_transform(df[features])
pca = PCA(n_components=2, random_state=42)
components = pca.fit_transform(X_scaled)
print("\\nVariance explained by first two PCs:",
      round(pca.explained_variance_ratio_.sum() * 100, 2), "%")

for class_id, name in enumerate(wine.target_names):
    mask = df.wine_class.to_numpy() == class_id
    plt.scatter(components[mask, 0], components[mask, 1], label=name, alpha=0.8)
plt.title("PCA Projection of Wine Samples")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend()
plt.tight_layout(); plt.savefig("outputs/pca_projection.png", dpi=180); plt.close()

model = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = model.fit_predict(X_scaled)
print("K-means silhouette score:", round(silhouette_score(X_scaled, cluster_labels), 3))
print("\\nMean feature values by class:\\n",
      df.groupby("wine_class_name")[features].mean().round(2))
