The purpose of this project is to understand how clustering algorithms work and apply to a real business problem.

I've applied K-Means algorithm and Hierarchical Clustering to find natural grouping in the dataset. This is useful for unsupervised learning, where labels are not available.

Customer dataset was acquired in the public website.

Some preprocessing steps include verifying for missing values, duplicates, normalizing the features using StandardScaler so every feature could fall within the same range. This helps to ensure one feature does not dominate over another.

K-means have shown to be innefective due to its model architecture, finding an optimal cluster requires on selecting the number of iterations. The elbow method was used to determine the number of k clusters.

Hierarchical clustering is often more robust than k-means since it avoids random initialization and can detect complex, non-spherical cluster structures.

Metrics for evaluation included Silhouette Score and Davies Bouldin Score.

Summary statistics were used to determine the label for each clusters.

Decision Trees were also employed afterwards to find what features drive the outcome, for better explainability to business stakeholders.

Unfortunately, the dataset was not complex enough to employ more advanced clustering algorithms (such as HDBSCAN), but it offers a practical experience.
