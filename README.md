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

# Customer Segmentation using Clustering Techniques

> A Machine Learning approach to identify and analyze key customer groups using unsupervised learning — enabling businesses to target their highest-value segments with precision.

**Fordham University** | Felipe Chen · Mykyta Shutov · Misha Semenov · Amareswar Doddi

## Table of Contents

- [Business Case](#-business-case)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Data Preprocessing](#-data-preprocessing)
- [Model Implementation](#-model-implementation)
- [Cluster Results & Profiles](#-cluster-results--profiles)
- [Marketing Strategy by Segment](#-marketing-strategy-by-segment)
- [Key Insights](#-key-insights)
- [Conclusion](#-conclusion)
- [Getting Started](#-getting-started)
- [Dependencies](#-dependencies)
- [Future Work](#-future-work)
- [Team](#-team)


## Business Case

In competitive retail markets, not all customers are created equal. Businesses that treat every customer identically leave significant revenue on the table — over-investing in low-value segments while under-serving high-value ones.

### The Core Problem

Most businesses have access to rich customer data but lack the tools to systematically extract actionable patterns from it. Without a structured segmentation framework:

- **Marketing budgets are diluted** across segments with vastly different value and responsiveness.
- **High-income, low-spend customers** — the biggest revenue opportunity — go unidentified and untargeted.
- **Loyal, high-spend customers** risk being overlooked because they don't need constant activation.
- **Retention strategies are generic**, reducing their effectiveness and increasing churn among engaged customers.

### The ML Solution

By applying unsupervised clustering to historical customer data, this project reveals **naturally occurring customer groups** based on income and spending behavior — without needing labeled data or predefined categories.

### Business Value

| Challenge | Without Segmentation | With ML Segmentation |

| Marketing targeting | Blanket campaigns | Segment-specific strategies |
| Budget allocation | Distributed equally | Weighted by segment value |
| High-income inactive customers | Invisible | Identified as "Opportunity Accounts" |
| Loyal low-income buyers | Underserved | Prioritized as "Devoted Fans" |
| Customer retention | Reactive | Proactive, persona-driven |

This analysis directly supports **targeted marketing**, **cross-selling**, **retention planning**, and **customer lifetime value (CLV) optimization**.


## Problem Statement

**Unsupervised Clustering:** Group customers based on their purchasing behavior, income, and demographic attributes — with no predefined labels — to identify which segments contribute most to sales and how each should be approached strategically.

- **Task Type:** Unsupervised Machine Learning (Clustering)
- **Primary Features:** Annual Income, Spending Score
- **Algorithms:** K-Means Clustering, Hierarchical (Agglomerative) Clustering
- **Evaluation Metrics:** Elbow Method (Inertia / WCSS), Silhouette Coefficient
- **Visualization:** Tableau, Matplotlib, Seaborn


## Dataset

**Source:** [Kaggle — Customer Segmentation Dataset](https://www.kaggle.com/)

**File:** `customer_segmentation_data.csv`

### Features

| Feature | Type | Description |

| `id` | Integer | Unique customer identifier *(dropped before modeling)* |
| `age` | Integer | Customer's age |
| `gender` | Categorical | Gender of the customer (`Male`, `Female`, `Other`) |
| `income` | Integer | Annual income of the customer (USD) |
| `spending_score` | Integer | Score representing spending habits (1–100) |
| `membership_years` | Integer | Years of membership with the company |
| `purchase_frequency` | Integer | Number of purchases per year |
| `preferred_category` | Categorical | Most frequently purchased product category |
| `last_purchase_amount` | Float | Amount spent in the most recent purchase |

### Key Observations from EDA

- **No simple linear relationship** between income and spending score — high earners don't necessarily spend more.
- **Most customers** fall in mid-to-high income ranges ($50K–$110K).
- **Spending scores are uniformly distributed** across 0–100, confirming multiple distinct behavioral types exist in the data.


## Project Workflow

```
1. Data Collection          →  Downloaded from Kaggle
        ↓
2. Data Preprocessing       →  Encoding, normalization, feature selection
        ↓
3. Feature Selection        →  Income + Spending Score chosen for clustering
        ↓
4. Model Building           →  K-Means (k=4) + Hierarchical Clustering (k=5)
        ↓
5. Evaluation               →  Elbow Method, Inertia, Silhouette Score
        ↓
6. Visualization            →  Tableau dashboards + Python plots
        ↓
7. Insight Generation       →  Segment profiles + targeted marketing strategies
```

## Exploratory Data Analysis

Three key visualizations were produced before clustering:

**Scatter Plot — Income vs Spending Score**
Spending behavior varies widely at every income level. No simple linear relationship exists, confirming that income alone cannot predict spending behavior and that clustering is needed.

**Bubble Chart — Income Distribution by Bins**
Customer income is concentrated between $50K and $110K. Very few customers fall at the extreme low or high ends, giving a natural mid-to-upper income skew.

**Histogram — Spending Score Distribution**
Spending scores spread roughly evenly across the 0–100 scale, with multiple behavioral clusters present — validating the unsupervised segmentation approach.

## Data Preprocessing

### 1. Dropped Identifier
The `id` column was removed as it carries no predictive information.

### 2. Missing Value & Duplicate Check
```python
df_dropped.isnull().sum()     # No missing values found
df_dropped.duplicated().sum() # No duplicates found
```

### 3. Categorical Encoding
`gender` and `preferred_category` were encoded using **One-Hot Encoding** (via `sklearn.preprocessing.OneHotEncoder`), producing binary indicator columns for each category:

```python
enc = OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False)
X = enc.fit_transform(df_dropped[["gender", "preferred_category"]])
```

Encoded categories for `preferred_category`: `Clothing`, `Electronics`, `Groceries`, `Home & Garden`, `Sports`.

### 4. Feature Selection for Clustering
After encoding, clustering was performed on **two primary numerical features** — `income` and `spending_score` — chosen for their direct relevance to customer value segmentation:

```python
numeric_cols = ['income', 'spending_score']
X_scaled = df_scaled[numeric_cols]
```

### 5. Standard Scaling
`StandardScaler` was applied to normalize the numeric features before clustering — essential for distance-based algorithms:

```python
scaler = StandardScaler()
df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
```

### 6. Pivot Table Analysis
After clustering, comprehensive pivot tables were built to compute per-cluster averages for all features — `age`, `income`, `spending_score`, `membership_years`, `purchase_frequency`, `last_purchase_amount` — enabling rich segment profiling.

## Model Implementation

### K-Means Clustering

**Optimal k Selection — Elbow Method:**
Inertia (WCSS) was computed for k = 1 through 10. The elbow point at k = 4 was selected as it produced the clearest inflection in the within-cluster sum of squares.

```python
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
```

**Final K-Means Model:**
```python
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=500, random_state=42)
y = kmeans.fit_predict(X_scaled)
```

**Silhouette Score:** `0.4190` — a reasonable score confirming meaningful cluster separation on this dataset.

**Why k-means++?** The `k-means++` initialization heuristic selects initial centroids that are spread apart, reducing the risk of poor convergence compared to random initialization.

### Hierarchical (Agglomerative) Clustering

**Dendrogram Construction:**
```python
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
```
The dendrogram's visual cut suggested **5 clusters** — capturing finer income gradations than K-Means.

**Final Hierarchical Model:**
```python
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
y_hc = hc.fit_predict(X_scaled)
```

**Silhouette Score:** `0.3741` — slightly lower than K-Means, as the 5-cluster solution is more granular and clusters are less compact by design.

**Ward Linkage** minimizes the total within-cluster variance at each merge step, producing compact and well-separated clusters suitable for this application.


### Model Comparison

| Algorithm | Clusters | Silhouette Score | Linkage / Init |

| **K-Means** | 4 | **0.4190** | k-means++ |
| Hierarchical | 5 | 0.3741 | Ward / Euclidean |

K-Means produced more compact, centroid-based clusters ideal for efficient high-level segmentation. Hierarchical Clustering provided finer granularity — separating the low-spending group into three sub-segments — offering deeper strategic insight into the lower half of the income distribution.

## Cluster Results & Profiles

### K-Means — 4 Clusters

Cluster centroids (standardized values):

| Cluster | Income (scaled) | Spending Score (scaled) | Profile Label |

| 0 | −0.93 | −0.92 | Low Income, Low Spend |
| 1 | +0.82 | −0.83 | High Income, Low Spend |
| 2 | −0.79 | +0.83 | Low Income, High Spend |
| 3 | +0.94 | +0.93 | High Income, High Spend |

**Key observation:** Average age is similar across all four clusters (early 40s), confirming that **age is not a meaningful differentiator** — spending behavior and income are the dominant segmentation axes.

### Hierarchical Clustering — 5 Clusters

| Cluster | Income (scaled) | Spending Score (scaled) | Profile Label |

| 0 | −0.82 | +0.80 | Low Income, High Spend |
| 1 | +0.22 | −0.69 | Mid Income, Low Spend |
| 2 | +0.99 | +0.92 | High Income, High Spend |
| 3 | −1.01 | −1.15 | Low Income, Low Spend |
| 4 | +1.32 | −0.82 | High Income, Low Spend |

**Strategic priority from HC:** Focus on Cluster 4 (high-income, low-spend — greatest untapped revenue), retain Cluster 2 (highest value), and nurture Cluster 1 (mid-income with conversion potential).

### Gender & Category Analysis
Cross-tabulation of clusters by gender and preferred category (Clothing, Electronics, Groceries, Home & Garden, Sports) revealed that **gender continues to influence purchasing preferences within clusters** — even among customers with similar income and spending profiles. This suggests that demographic sub-targeting within each segment can further improve campaign performance.

## Marketing Strategy by Segment

The four K-Means segments were translated into business personas with actionable strategies:

### Devoted Fans — High Spend / Low Income
*The largest customer segment. High engagement despite lower income.*

| Strategy | Action |

| **Loyalty Programs** | Reward frequent purchases with points, discounts, and exclusive access |
| **Value-Based Messaging** | Emphasize quality, durability, and long-term value |
| **Community Building** | Brand communities and user-generated content campaigns |
| **Flexible Payment** | Installment plans and seasonal promotions to support purchasing power |

---

### Power Customers — High Spend / High Income
*Premium buyers. Highest revenue per customer with strong profit margins.*

| Strategy | Action |

| **VIP Treatment** | Exclusive early access to new products, private sales, dedicated service |
| **Premium Product Lines** | Curated high-end collections for this segment |
| **Personalized Experiences** | One-on-one consultations and white-glove service |
| **Strategic Partnerships** | Luxury co-branded experiences and collaborations |


### Emerging Relationships — Low Spend / Low Income
*Early-stage customers. Price-sensitive with significant growth potential.*

| Strategy | Action |

| **Entry-Level Products** | Affordable starter products to build initial purchase habits |
| **Educational Content** | Tutorials and product education to build trust |
| **Gradual Engagement** | Low-cost email and social media touchpoints |
| **Referral Incentives** | Word-of-mouth growth through friend referral programs |

### Opportunity Accounts — Low Spend / High Income
*Affluent but inactive. Greatest untapped revenue potential.*

| Strategy | Action |

| **Targeted Promotions** | Compelling re-engagement offers for dormant high-value prospects |
| **Premium Positioning** | Showcase products matching their income level and lifestyle |
| **Competitive Analysis** | Understand what competitors offer; develop superior value propositions |
| **Direct Outreach** | Personal sales and account management to identify and remove barriers |

## Key Insights

**1. Spending Score is the dominant segmentation factor.** It divides customers into two stable, consistent groups — high spenders and low spenders — more reliably than income. Income adds a second dimension but is secondary in predictive power.

**2. Income and spending are decoupled.** The absence of a linear relationship between income and spending score is the central finding. High earners are not automatically high spenders — the "Opportunity Accounts" group proves this at scale.

**3. Age is not a meaningful segmentation variable.** All four K-Means clusters show nearly identical average ages (early 40s). Campaigns should not be age-targeted; income and spending behavior are the effective levers.

**4. Gender influences category preferences within clusters.** Even within the same income-spending tier, purchasing category preferences vary by gender — suggesting two-dimensional targeting (segment × gender) for category-specific campaigns.

**5. The low-spending segment is the largest.** The strategic priority is converting low-spend customers into higher-spend ones through brand loyalty, community, and engagement strategies — not just acquiring new customers.

## Conclusion

Both clustering algorithms successfully identified distinct customer segments:

- **K-Means (k=4)** produced compact, centroid-based clusters ideal for straightforward, high-level marketing segmentation with a Silhouette Score of 0.419.
- **Hierarchical Clustering (k=5)** revealed finer groupings — particularly splitting the low-spending tier into three income-based sub-segments — enabling more nuanced targeting.

Together, these models provide a **scalable, data-driven foundation** for customer segmentation that can be refreshed periodically as new transaction data becomes available.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/customer-segmentation-clustering.git
cd customer-segmentation-clustering
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download `customer_segmentation_data.csv` from [Kaggle](https://www.kaggle.com/) and place it in the project root.

### 4. Run the Notebook

```bash
jupyter notebook ML_Project_Clustering.ipynb
```

### 5. Explore Outputs

Two CSV files are generated for Tableau or further analysis:
- `Kmeans_Clusters.csv` — original data with K-Means cluster labels
- `HC_Clusters.csv` — original data with Hierarchical Clustering labels

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
jupyter
```

Install all at once:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

## Project Structure

```
customer-segmentation-clustering/
│
├── ML_Project_Clustering.ipynb       # Main analysis and modelling notebook
├── customer_segmentation_data.csv    # Raw dataset (download from Kaggle)
├── Kmeans_Clusters.csv               # Output: K-Means cluster assignments
├── HC_Clusters.csv                   # Output: Hierarchical cluster assignments
├── README.md                         # Project documentation
└── requirements.txt                  # Python dependencies
```

## Future Work

- [ ] **A/B Testing** — run controlled experiments exposing different segments to targeted promotions and measure lift in conversion and spend
- [ ] **Enrich features** — incorporate purchase history, recency, browsing behavior, geographic location, and sentiment scores for deeper segmentation
- [ ] **DBSCAN / Gaussian Mixture Models** — explore density-based and probabilistic clustering for non-spherical cluster shapes
- [ ] **Real-time dashboard** — build a live Tableau or Power BI dashboard connected to incoming transaction data
- [ ] **Cluster stability testing** — use bootstrap resampling to assess how stable cluster assignments are across different data samples
- [ ] **RFM Analysis** — layer Recency-Frequency-Monetary metrics on top of clusters for direct marketing scoring


## Team

**Fordham University** — Machine Learning Course Project

| Name | Contribution |

| Felipe Chen | Clustering implementation, EDA, preprocessing |
| Mykyta Shutov | Hierarchical clustering, dendrogram analysis |
| Misha Semenov | Visualization, segment profiling |
| Amareswar Doddi | Strategic insights, marketing recommendations |


## References

- Dataset: [Kaggle — Customer Segmentation](https://www.kaggle.com/)
- Tools: Python (`scikit-learn`, `pandas`, `seaborn`, `scipy`), Tableau
- Data Science from Provost.

> *Built with Python and scikit-learn*
