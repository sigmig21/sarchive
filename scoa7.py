import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min

# --- Parameters for Clustering and PSO ---
K = 5  # Number of clusters
N_FEATURES = 2 # Annual Income and Spending Score
N_PARTICLES = 20 # Number of particles in the swarm
MAX_ITER = 100 # Maximum number of iterations
W = 0.7  # Inertia weight
C1 = 1.5 # Cognitive coefficient (personal best)
C2 = 1.5 # Social coefficient (global best)

# --- 1. Load and Preprocess Data ---

# Load the dataset
df = pd.read_csv("SCOA_A7.csv")

# Select features for clustering: 'Annual Income (k$)' and 'Spending Score (1-100)'
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Scale the data: Standardization is essential for distance-based clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Data shape after scaling: {X_scaled.shape}")

def fitness_function(position, data, k):
    """
    Calculates the WCSS (Inertia) for a given set of centroids.

    Args:
        position (np.array): A 1D array representing the flattened centroids (k * n_features).
        data (np.array): The scaled data points.
        k (int): The number of clusters.

    Returns:
        float: The WCSS value (Inertia).
    """
    n_features = data.shape[1]
    # Reshape the 1D position vector into a 2D array of centroids (K x N_FEATURES)
    centroids = position.reshape(k, n_features)

    # Calculate the distance from each data point to every centroid and
    # find the index of the closest centroid (cluster assignment)
    # The second return value is the minimum distance squared.
    distances = np.sum((data[:, np.newaxis, :] - centroids[np.newaxis, :, :])**2, axis=2)
    min_distances_sq = np.min(distances, axis=1)

    # WCSS is the sum of the squared distances from each point to its assigned centroid
    wcss = np.sum(min_distances_sq)
    return wcss

def pso_clustering(data, k, n_particles, max_iter, w, c1, c2):
    """
    Performs K-Means clustering using Particle Swarm Optimization.

    Returns:
        tuple: (best_centroids, gbest_wcss)
    """
    n_features = data.shape[1]
    # The total number of dimensions for the particle's position vector
    n_dimensions = k * n_features

    # 1. Initialize Particles and Velocities
    # Initialize particle positions (centroids) randomly within the data's bounds.
    min_bounds = np.min(data, axis=0)
    max_bounds = np.max(data, axis=0)

    # Reshape min/max for broadcasting: (1, N_FEATURES)
    min_b = min_bounds[np.newaxis, :]
    max_b = max_bounds[np.newaxis, :]
    
    # Initialize positions (N_PARTICLES, K, N_FEATURES)
    # Generate random numbers and scale them to the feature range
    positions = np.random.rand(n_particles, k, n_features) * (max_b - min_b) + min_b
    
    # Flatten positions to (N_PARTICLES, N_DIMENSIONS)
    positions = positions.reshape(n_particles, n_dimensions)
    
    # Initialize velocities to zero (N_PARTICLES, N_DIMENSIONS)
    velocities = np.zeros((n_particles, n_dimensions))

    # 2. Initialize Personal Best (pbest) and Global Best (gbest)
    pbest_positions = positions.copy()
    pbest_wcss = np.array([fitness_function(p, data, k) for p in pbest_positions])

    # Find initial global best (gbest)
    gbest_idx = np.argmin(pbest_wcss)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_wcss = pbest_wcss[gbest_idx]

    # --- 3. Main PSO Loop ---
    for iteration in range(max_iter):
        for i in range(n_particles):
            # Generate random coefficients for C1 and C2
            r1 = np.random.rand(n_dimensions)
            r2 = np.random.rand(n_dimensions)

            # --- Update Velocity ---
            # 1. Inertia component
            inertia = w * velocities[i]
            # 2. Cognitive component (moves towards pbest)
            cognitive = c1 * r1 * (pbest_positions[i] - positions[i])
            # 3. Social component (moves towards gbest)
            social = c2 * r2 * (gbest_position - positions[i])

            velocities[i] = inertia + cognitive + social

            # --- Update Position ---
            positions[i] += velocities[i]

            # --- Evaluate Fitness (WCSS) ---
            current_wcss = fitness_function(positions[i], data, k)

            # --- Update Personal Best (pbest) ---
            if current_wcss < pbest_wcss[i]:
                pbest_wcss[i] = current_wcss
                pbest_positions[i] = positions[i].copy()

            # --- Update Global Best (gbest) ---
            if current_wcss < gbest_wcss:
                gbest_wcss = current_wcss
                gbest_position = positions[i].copy()
                
        print(f"Iteration {iteration+1}/{max_iter}: Best WCSS = {gbest_wcss:.2f}")

    # Return the final optimal centroids and the best WCSS
    best_centroids = gbest_position.reshape(k, n_features)
    return best_centroids, gbest_wcss

# --- 4. Execution ---

best_centroids_scaled, final_wcss = pso_clustering(
    X_scaled, K, N_PARTICLES, MAX_ITER, W, C1, C2
)

# Reverse the scaling to get the centroids in the original units (k$ and 1-100 score)
best_centroids_original = scaler.inverse_transform(best_centroids_scaled)

# Assign clusters to all data points using the final optimal centroids
# Get the cluster labels (the closest centroid index)
cluster_labels, _ = pairwise_distances_argmin_min(X_scaled, best_centroids_scaled)

# Add the cluster labels to the original DataFrame
df['Cluster'] = cluster_labels

# --- 5. Display Final Results ---

print("\n--- PSO Clustering Results ---")
print(f"Final Minimum WCSS achieved: {final_wcss:.2f}")
print("\nOptimal Centroids (in k$ and 1-100 Score):")
centroids_df = pd.DataFrame(best_centroids_original, columns=['Annual Income (k$)', 'Spending Score (1-100)'])
print(centroids_df)

print("\nCluster Sizes:")
print(df['Cluster'].value_counts().sort_index())

# Save the results to a CSV for analysis
df.to_csv("PSO_Clustering_Results.csv", index=False)


"""

## 🎯 **Overall Significance of the Code**

This code performs **customer segmentation** (clustering) using **Particle Swarm Optimization (PSO)** instead of the traditional K-Means algorithm.

* **Dataset used:** Mall Customer Data (`SCOA_A7.csv`)
* **Features:**

  * *Annual Income (k$)*
  * *Spending Score (1–100)*
* **Goal:** Group customers with similar income and spending patterns into 5 clusters (`K = 5`).

---

## 🧠 **Conceptual Idea Behind PSO in Clustering**

Traditional **K-Means** uses random centroids and iterative updates based on mean positions.
However, K-Means can **get stuck in local minima** due to random initialization.

👉 PSO is a **metaheuristic optimization technique** inspired by the **social behavior of birds flocking or fish schooling**.
Here, each *particle* represents a **possible solution** (set of cluster centroids), and the swarm collectively searches for the best clustering by minimizing the **WCSS (Within Cluster Sum of Squares)**.

---

## ⚙️ **Implementation Steps Explained**

### **1. Data Preprocessing**

```python
df = pd.read_csv("SCOA_A7.csv")
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

* The data is read and scaled to standardize features (since income and score have different scales).
* Scaling ensures fair distance computation in clustering.

---

### **2. Fitness Function**

```python
def fitness_function(position, data, k):
```

* Each particle encodes *k centroids* → `position` = flattened array of centroids.
* **Fitness** = **WCSS** (Within Cluster Sum of Squares).
  [
  WCSS = \sum_{i=1}^{n} \min_{j=1}^{k} ||x_i - c_j||^2
  ]
* Lower WCSS means better clustering (data points closer to their centroids).

So, PSO tries to **minimize this WCSS**.

---

### **3. PSO Initialization**

```python
positions = np.random.rand(n_particles, k, n_features)
velocities = np.zeros((n_particles, n_dimensions))
```

* Each particle = one set of possible centroids (shape: K × features).
* Velocities are initialized to 0.
* Positions are initialized randomly within data bounds.

---

### **4. Personal Best (pbest) & Global Best (gbest)**

Each particle remembers:

* Its best known position (lowest WCSS) → **pbest**
* The overall best position in the swarm → **gbest**

Initially:

```python
pbest_positions = positions.copy()
pbest_wcss = np.array([fitness_function(p, data, k) for p in positions])
gbest_idx = np.argmin(pbest_wcss)
```

---

### **5. Main PSO Loop**

```python
for iteration in range(max_iter):
```

At every iteration:

#### (a) **Velocity Update Equation**

[
v_i = wv_i + c_1r_1(pbest_i - x_i) + c_2r_2(gbest - x_i)
]

* **w**: inertia (controls exploration/exploitation)
* **c1**: cognitive component (particle’s own experience)
* **c2**: social component (influence of the best particle)
* **r1, r2**: random values between 0 and 1

#### (b) **Position Update**

[
x_i = x_i + v_i
]
→ Centroids move in the direction guided by velocity.

#### (c) **Fitness Evaluation**

After updating positions, compute WCSS again.
If a new position gives better WCSS:

* Update `pbest`
* Possibly update `gbest`

---

### **6. Optimization Output**

At each iteration, best WCSS is printed:

```
Iteration 1/100: Best WCSS = 124.18
...
Iteration 100/100: Best WCSS = 65.57
```

This shows **progressive improvement** — PSO gradually finds better centroids.

---

## 📊 **Final Results and Interpretation**

### ✅ **Final Minimum WCSS = 65.57**

* Indicates the optimal (minimum) total within-cluster variance.
* Lower WCSS = tighter clusters → good result.

### ✅ **Optimal Centroids**

```
   Annual Income (k$)  Spending Score (1-100)
0           86.56               82.12
1           88.17               17.09
2           25.71               79.37
3           55.30               49.51
4           26.32               20.92
```

These centroids represent **cluster centers** (income vs spending score).

Interpretation:

| Cluster | Annual Income | Spending Score | Customer Type                                            |
| ------- | ------------- | -------------- | -------------------------------------------------------- |
| 0       | High          | High           | Premium spenders                                         |
| 1       | High          | Low            | Wealthy but frugal                                       |
| 2       | Low           | High           | Low income, high spenders (potentially impulsive buyers) |
| 3       | Medium        | Medium         | Average customers                                        |
| 4       | Low           | Low            | Budget-conscious buyers                                  |

---

### ✅ **Cluster Sizes**

```
Cluster
0    39
1    35
2    22
3    81
4    23
```

Indicates number of customers in each cluster.
Cluster 3 is the largest (81 customers) → most people fall in the *average* category.

---

## 🧩 **Significance of Using PSO for Clustering**

| Aspect         | K-Means                       | PSO-based Clustering                  |
| -------------- | ----------------------------- | ------------------------------------- |
| Initialization | Random centroids              | Population of candidate centroids     |
| Optimization   | Gradient-like local updates   | Global search with swarm intelligence |
| Convergence    | May get stuck in local minima | More likely to find global optimum    |
| Speed          | Fast                          | Slower (but more accurate)            |
| Result         | Depends on initial seeds      | Robust, stable results                |

---

## 🗣️ **Viva Key Points (Quick Answers)**

**Q1. What is the objective function used here?**
→ The **fitness function** is WCSS — we aim to **minimize intra-cluster distance**.

**Q2. What do pbest and gbest mean?**
→ *pbest*: best position found by a particle.
→ *gbest*: best position found by the entire swarm.

**Q3. What are W, C1, and C2?**
→ PSO hyperparameters:

* **W (Inertia)**: controls momentum.
* **C1 (Cognitive)**: particle’s own learning.
* **C2 (Social)**: learning from others.

**Q4. How does PSO improve clustering compared to K-Means?**
→ It avoids local minima, uses population-based search, and gives more stable clustering results.

**Q5. What does the final WCSS tell you?**
→ It measures total intra-cluster variation. Smaller WCSS = better clustering.

---

## 🧮 **Summary of What’s Implemented**

| Step                         | Description                                   |
| ---------------------------- | --------------------------------------------- |
| 1️⃣ Data loading and scaling | Prepare income & spending data for clustering |
| 2️⃣ Fitness function         | Defines clustering quality (WCSS)             |
| 3️⃣ Swarm initialization     | Create random centroids and velocities        |
| 4️⃣ Iterative optimization   | Update particles’ positions and velocities    |
| 5️⃣ Best centroid selection  | Choose centroids with minimum WCSS            |
| 6️⃣ Assign clusters          | Label data points based on nearest centroid   |
| 7️⃣ Output results           | Display final clusters, centroids, and WCSS   |

---
"""