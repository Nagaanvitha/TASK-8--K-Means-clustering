# TASK-8--K-Means-clustering
 Clustering &amp; unsupervised learning- Mall Customer Segmentation

# K-Means Clustering Overview
K-Means aims to partition your data into K distinct, non-overlapping clusters based on similarity. It minimizes the within-cluster sum of squares (WCSS) — i.e., how close data points in the same cluster are to the centroid of that cluster.

# In Your Code: Key Steps of K-Means
python
Copy
Edit
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    y_kmeans = kmeans.fit_predict(data[['Annual Income (k$)', 'Spending Score (1-100)']])
    data['KMeans_Labels'] = y_kmeans
# Step-by-Step Breakdown:
Loop Over Possible Cluster Counts (1 to 10):

You're trying multiple values of K to find the optimal number of clusters using the Elbow Method.

Create the KMeans Object:

python
Copy
Edit
kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
n_clusters=i: The current number of clusters being tested.

init='k-means++': A smart initialization that helps speed up convergence.

n_init=10: Runs the algorithm 10 times and picks the best one (lowest WCSS).

random_state=42: Ensures reproducibility.

Fit the Model:

python
Copy
Edit
kmeans.fit(X)
The model learns cluster centers (centroids) and assigns points to their nearest one.

Store WCSS:

python
Copy
Edit
wcss.append(kmeans.inertia_)
WCSS (Within-Cluster Sum of Squares) is a measure of compactness of clusters. Lower WCSS is better.

Predict Cluster Labels:

python
Copy
Edit
y_kmeans = kmeans.fit_predict(data[['Annual Income (k$)', 'Spending Score (1-100)']])
data['KMeans_Labels'] = y_kmeans
Assigns each point to a cluster.

Stores labels in data['KMeans_Labels'].

# Note: You're using data[['Annual Income (k$)', 'Spending Score (1-100)']] instead of X (which is the scaled version). This may slightly skew results unless intentional.

# Elbow Method
Later, you plot the WCSS values:

python
Copy
Edit
plt.plot(range(1, 11), wcss, marker='o')
The “elbow point” (where the WCSS drops sharply then levels off) is considered the best K.

# What Happens Internally in K-Means:
For a given K:

Initialize K centroids.

Assign each point to the nearest centroid.

Recalculate the centroids (mean of points in each cluster).

Repeat steps 2–3 until assignments don’t change or max iterations reached.


# Concept of the Elbow Method
The Elbow Method helps you pick the right number of clusters by plotting:

X-axis: Number of clusters K (from 1 to 10 in your case)

Y-axis: WCSS (Within-Cluster Sum of Squares), which measures how compact each cluster is.

As K increases, WCSS decreases (clusters get tighter), but after a certain point, the improvement diminishes — this is the "elbow point."

# In Your Code:
python
Copy
Edit
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
kmeans.inertia_ gives the WCSS for the current number of clusters.

You store these in the list wcss.

Then you plot the values:

python
Copy
Edit
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='purple')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal K')
plt.show()
This plot helps you visually inspect the "elbow" — the point where the curve starts to bend and flatten. The corresponding K is your optimal number of clusters.

# Example Interpretation
If your elbow plot looks like this (conceptually):

mathematica
Copy
Edit
WCSS
│
│     *
│    * 
│   *   
│  *    
│ *
│*
└─────────────────────▶ K (Number of Clusters)
     1   2   3   4   5   6   ...
The elbow is at K = 5, suggesting that 5 clusters might be the optimal choice — which aligns with what you used in your Agglomerative Clustering too.


# Limitations of K-Means in Your Case
1. Assumes Spherical and Equal-Sized Clusters
K-Means works best when clusters are circular (convex) and roughly the same size.

In your customer data (Annual Income vs. Spending Score), some groups may be elongated or of different densities, which K-Means struggles to separate properly.

2. Sensitive to Initial Centroid Placement
Even though you're using init='k-means++' (which improves initialization), K-Means can still converge to local minima.

Running it multiple times (n_init=10) helps, but results can still vary slightly.

3. Requires Predefined Number of Clusters (K)
You use the Elbow Method to choose K, but this approach is subjective and doesn't guarantee the best K.

Choosing the wrong K can lead to poor clustering.

4. Sensitive to Outliers
K-Means uses mean positions for centroids, so outliers can pull centroids far from their "true" cluster center.

This is especially relevant if customer data contains extreme spenders or earners.

5. Fails with Non-Convex Shapes
If your data forms non-linear clusters, K-Means can’t capture the boundaries.

For example, clusters shaped like moons or rings will be poorly modeled.

6. Distance Metric Limitations
K-Means uses Euclidean distance, which may not capture true similarity if features aren't appropriately scaled (you did scale them using StandardScaler, which is good).

7. Assigns Every Point to a Cluster
Unlike DBSCAN, which can label noise points (-1), K-Means forces every point into a cluster, even if it doesn't fit well.

# Why It Still Worked Reasonably in Your Case
Your selected features — Annual Income and Spending Score — tend to naturally form well-separated, roughly convex clusters, so K-Means can perform decently. But for more complex datasets (e.g. with more features or non-linear relationships), these limitations become more problematic.


# Why Initialization Matters
K-Means starts by randomly selecting K initial centroids. From there, it:

Assigns points to the nearest centroid.

Recomputes centroids based on assigned points.

Repeats until convergence.

If the initial centroids are poorly chosen:

The algorithm can converge to suboptimal clusters (i.e., local minima).

You may get different results each time you run the code.

Clusters might not reflect the true underlying structure.

# In Your Code: Controlled Initialization
python
Copy
Edit
kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
Here's how each parameter helps:

# init='k-means++'
Improves initialization by spreading out the initial centroids rather than picking them randomly.

Leads to faster convergence and better clustering results.

Reduces the risk of poor local minima.

# n_init=10
Runs the K-Means algorithm 10 times with different initializations, and picks the best one based on lowest WCSS.

Provides more stability and accuracy.

# random_state=42
Fixes the random seed so that results are reproducible — crucial for debugging or comparing models.

# If You Didn’t Use These Settings...
If you had:

python
Copy
Edit
KMeans(n_clusters=5)
...K-Means would:

Use random initialization.

Run only once (n_init=1 by default in older versions).

Give inconsistent and possibly worse cluster quality.

# Potential Symptoms of Poor Initialization
Highly imbalanced clusters.

Non-reproducible results across runs.

High WCSS or low silhouette scores.

Visually poor separation in the scatter plot.

# Tip for Robust Results
Even with k-means++, it’s still worth experimenting with:

Larger n_init (e.g. 20 or 50 for production).

Silhouette scores and visual validation.

Alternative clustering methods (e.g. DBSCAN, Agglomerative) as a cross-check.

# In Your Code:
python
Copy
Edit
kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
kmeans.fit(X)
wcss.append(kmeans.inertia_)
kmeans.inertia_ stores the WCSS (within-cluster sum of squares) for the current value of K.

You're collecting this value in the wcss list to plot the Elbow Method.

# What Inertia Tells You
Lower inertia = tighter, more compact clusters.

Higher inertia = more spread-out points (worse fit).

As K increases, inertia always decreases, because more clusters = more centroids = closer distances.

# Limitations of Inertia:
It always decreases as K increases, so you can't use inertia alone to pick the "best" K — hence the Elbow Method.

It assumes spherical clusters and Euclidean distance, which may not suit all data types.

# Summary
Inertia in your code helps you:

Quantify clustering quality at each value of K.

Build the Elbow curve to find the optimal number of clusters.


# Definition of Silhouette Score
The Silhouette Score measures how similar a data point is to its own cluster (cohesion) compared to other clusters (separation). It ranges from -1 to +1:
s(i)= b(i)−a(i)/max(a(i),b(i))
​Where:
-a(i) = average distance from point 
-i to all other points in its own cluster
-b(i) = average distance from point 
-i to all points in the nearest other cluster

# Interpretation
+1: Point is well-clustered

0: Point is on the decision boundary

-1: Point is likely in the wrong cluster

# In Your Code:
python
Copy
Edit
kmeans_silhouette = silhouette_score(data[['Annual Income (k$)', 'Spending Score (1-100)']], y_kmeans)
agglomerative_silhouette = silhouette_score(data[['Annual Income (k$)', 'Spending Score (1-100)']], y_agglo)
dbscan_silhouette = silhouette_score(data[['Annual Income (k$)', 'Spending Score (1-100)']], y)
You're calculating the average Silhouette Score for each clustering algorithm:

K-Means

Agglomerative Clustering

DBSCAN

Each score tells you how well the points were assigned to clusters overall.

# Caution with DBSCAN:
DBSCAN assigns some points as noise (-1), which can distort the silhouette score.

A better approach:

python
Copy
Edit
from sklearn.metrics import silhouette_score
mask = y != -1
dbscan_silhouette = silhouette_score(X[mask], y[mask])
# Why Use Silhouette Score (vs. Inertia)?
Metric	What it Measures	Good For
Inertia	Compactness within clusters	Elbow Method
Silhouette	Compactness and separation	Quality of Clusters

# When Is It Useful?
When comparing multiple clustering algorithms (as you did).

When picking the best K for K-Means (higher silhouette is better).

When testing whether clusters are well-separated.


# 1. Elbow Method (Used in Your Code)
How it works: Plot the Within-Cluster Sum of Squares (WCSS) for different values of K.

Goal: Find the "elbow" point where WCSS stops decreasing significantly.

In your code:

python
Copy
Edit
wcss.append(kmeans.inertia_)
Then you plot WCSS vs. K.

# Limitation: The elbow can sometimes be hard to spot or ambiguous.

# 2. Silhouette Score (Also Used in Your Code)
How it works: Measures how well a point fits into its cluster compared to others.

Range: -1 to +1; higher is better.

In your code:

python
Copy
Edit
silhouette_score(data[['Annual Income (k$)', 'Spending Score (1-100)']], y_kmeans)
Try different K values and pick the one with the highest score.

# Limitation: Computationally heavier than WCSS if your dataset is large.

# 3. Gap Statistic (Not Used Yet)
Compares WCSS for your clustering with WCSS from random, uniformly distributed data.

More statistically robust, but complex to implement and slower.

# 4. Visual Inspection of Cluster Plots
After running clustering for different K values, visually inspect the plots (like your Plotly scatter plots).

Ask yourself: Are the clusters clearly separated and meaningful?

# Practical Tip (Automate Selection):
You can combine both Elbow and Silhouette into a loop:

python
Copy
Edit
from sklearn.metrics import silhouette_score

wcss = []
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    score = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(score)
Then plot both to compare.

🔄# Summary Table
Method	             What it Measures	      Your Code	         Recommended
Elbow Method	       WCSS vs. K	                ✅	             Yes
Silhouette Score Compactness & separation  	✅	             Yes
Gap Statistic	Statistical null comparison	❌                Optional
Visual Inspection   Human interpretability	✅                 Always


# Fundamental Difference
Feature	Clustering	                                      Classification
Type of Learning	                 Unsupervised Learning	                                      Supervised Learning
Goal	                 Group similar data points without labels	                         Assign predefined labels to data points
Data Labels	                No prior labels available	                                      Requires labeled training data
Output	                         Cluster IDs (e.g., 0, 1, 2...)	                                     Class labels (e.g., “High spender”)
Example	                  Segmenting mall customers based on behavior	                   Predicting if a customer will churn

# In Your Code — You're Doing Clustering:
# Unsupervised Learning:
You're using K-Means, Agglomerative, and DBSCAN to:

Group customers based on patterns in their Annual Income and Spending Score.

No label like "Luxury Shopper" or "Budget Buyer" is provided — the algorithm finds patterns on its own.

# Not Classification Because:
You're not training on labeled examples (e.g., you don't have a column like CustomerType to predict).

There’s no “target variable” — you’re discovering structure in the data, not predicting predefined categories.

# Example to Clarify:
Imagine you have a dataset like this:

Customer	Annual Income	Spending Score	Customer Type
  001	             80k	      90	Luxury Shopper
  002	             20k	      20	Budget Buyer
...	...	...	...

Classification: You would train a model to predict Customer Type based on income and score.

Clustering: You don’t use the Customer Type. Instead, you let the algorithm discover natural groupings, which might align (or not) with "Luxury Shopper" or "Budget Buyer".

# Final Thought
Clustering = Discovering structure

Classification = Predicting structure

In your customer segmentation notebook, clustering helps discover customer segments without prior assumptions — ideal for marketing, customer profiling, or exploratory analysis.




