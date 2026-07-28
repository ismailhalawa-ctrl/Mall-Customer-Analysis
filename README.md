# 🛍️ Mall Customer Segmentation & Behavioral Analytics

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 Project Overview
This project applies **Unsupervised Machine Learning** techniques to perform customer segmentation for a supermarket chain. By analyzing customer demographics and spending behaviors, the business can identify target customer groups (e.g., high-income high-spenders vs. low-efficiency spenders) and tailor marketing strategies accordingly.

The dataset contains **200 customer profiles** with features including Age, Gender, Annual Income, and Spending Score.

---

## 🛠️ Key Pipeline & Features

1. **Feature Engineering**: Created custom metrics to better capture customer dynamics:
   * `Spending Per Age`: Measures spending tendencies relative to age.
   * `Spending Efficiency`: Ratio of Spending Score to Annual Income.
2. **Preprocessing & Scaling**: Encoded categorical variables (`Gender`) and applied `StandardScaler` to normalize numerical features.
3. **Dimensionality Reduction**:
   * **PCA**: Reduced data dimensions while retaining **>90% variance** with 3 principal components (and ~95% with 4 components).
   * **t-SNE**: Mapped non-linear relationships into 2D space for cluster verification.
4. **Clustering & Sensitivity Analysis**: Compared **K-Means**, **MiniBatch K-Means**, and **DBSCAN** across raw and PCA-reduced feature spaces.

---

## 📊 Model Comparison & Results

We evaluated the clustering models based on **Inertia** (cluster compactness), **Silhouette Score** (cluster separation), and **Execution Speed**:

| Clustering Algorithm | Space | Clusters (k) | Inertia | Silhouette Score | Execution Time (s) | Observations |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **K-Means (Optimal)** | **PCA-Reduced** | **5** | **222.05** | **0.4115** | **~0.0069s** | **Best cluster separation and structure.** |
| **K-Means** | Full Scaled | 5 | 332.14 | 0.3222 | ~0.0075s | Good, but higher inertia due to full dimensions. |
| **MiniBatch K-Means** | PCA-Reduced | 5 | 270.77 | 0.3500 | ~0.0107s | Slightly noisier boundaries; overhead on small data. |
| **DBSCAN** | PCA-Reduced | 5 | — | 0.2100 | ~0.0063s | Struggled with density variance (12 noise points). |

> **Key Takeaway**: **K-Means (k=5) on PCA-reduced data** proved to be the superior pipeline, providing the most distinct customer groups and highest Silhouette Score (0.4115).

---

## 📈 Key Insights & Business Impact

* **High-Value Target Segment**: Customers with high annual income and high spending scores were clearly segmented, allowing the marketing team to focus VIP loyalty programs on them.
* **Dimensionality Advantage**: Reducing features via PCA improved the Silhouette Score from 0.3222 to 0.4115, proving that removing feature redundancy enhances clustering quality.

---

## 🚀 How to Run

1. **Clone the Repository**:
```bash
git clone [https://github.com/ismailhalawa-ctrl/Mall-Customer-Analysis.git](https://github.com/ismailhalawa-ctrl/Mall-Customer-Analysis.git)
cd Mall-Customer-Analysis
```

2. **Install Dependencies**:
```bash
pip install numpy pandas matplotlib scikit-learn
```

3. **Run the Notebook**:
Open the Jupyter Notebook or Google Colab file and execute all cells sequentially.

---

## 📄 Dataset
* **Source**: Mall Customer Segmentation Dataset
* **Instances**: 200
* **Features**: `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)`
