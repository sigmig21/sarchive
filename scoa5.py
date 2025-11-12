import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense

# --- Configuration ---
# File path for the uploaded dataset
FILE_PATH = 'SCOA_A5.csv'
# The CSV contains multiple stocks. We filter for one to predict future movement.
STOCK_TICKER = 'AAL'

# Step 1: Load stock data from CSV
try:
    data = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}")
    exit()

# Filter data for the selected stock ticker
df = data[data['Name'] == STOCK_TICKER].copy()

# Ensure the data is sorted by date for correct shifting
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Create the prediction target: 1 if the next day's close is higher, 0 otherwise.
# Note: The column names are in lowercase in the CSV (e.g., 'close').
df['Target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

# Drop the last row, as its target cannot be calculated (Target will be NaN)
df = df.dropna()

print(f"Data loaded for {STOCK_TICKER}. Total records: {len(df)}")


# Step 2: Feature selection and scaling
# Use the same features as the original script, adjusted for lowercase column names.
features = df[['open', 'high', 'low', 'close', 'volume']]
scaler = MinMaxScaler()
X = scaler.fit_transform(features)
y = df['Target'].values.astype(np.int32) # Ensure target is integer type

# Step 3: Train-test split
# Using shuffle=False is critical for time-series data to maintain chronological order.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4: Build and train ANN model
# Input dimension is 5 (open, high, low, close, volume)
model = Sequential()
model.add(Dense(64, input_dim=5, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Training model...")
# Training with reduced verbosity for cleaner output
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)
print("Training complete.")

# Step 5: Evaluate model
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = (y_pred_proba > 0.5).astype("int32")

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\n--- Model Evaluation for {STOCK_TICKER} ---")
print(f"Test Accuracy: {accuracy:.4f}")
print("Confusion Matrix (True Negatives, False Positives, False Negatives, True Positives):")
print(conf_matrix)

# Interpretation of the confusion matrix (optional but helpful)
tn, fp, fn, tp = conf_matrix.ravel()
print(f"\n- True Positives (Correctly predicted UP): {tp}")
print(f"- True Negatives (Correctly predicted DOWN): {tn}")
print(f"- False Positives (Predicted UP, but went DOWN): {fp}")
print(f"- False Negatives (Predicted DOWN, but went UP): {fn}")


"""
---

### 🔍 **1. What the Output Means**

* **Accuracy: 0.5079 (≈ 50.8%)**

  * The model is barely better than random guessing (since stock movements are ~50–50 up/down daily).
* **Confusion Matrix:**

  ```
  [[117   6]
   [118  11]]
  ```

  * **True Negatives (117):** Correctly predicted price *down* days.
  * **False Positives (6):** Predicted *up*, but price actually went *down*.
  * **False Negatives (118):** Predicted *down*, but price actually went *up*.
  * **True Positives (11):** Correctly predicted *up* days.

👉 The model is biased toward predicting *“down”* days — it’s getting most “down” days correct but missing many “up” movements.

---

### ⚙️ **2. Why Accuracy is Low**

Several reasons are typical for this type of stock dataset:

1. **Stock prices are highly noisy** — small daily fluctuations are hard to predict.
2. **Input features (OHLCV)** may not carry enough predictive signal.
3. **Time-series nature** — your train/test split is correct (`shuffle=False`), but the ANN doesn’t capture temporal dependencies (it treats each day as independent).
4. **Target imbalance** — there might be more “down” days than “up” days, causing a class bias.

---

## 🧠 Conceptual Explanation — *What the Model is Doing*

### 🎯 **Goal**

Predict **whether tomorrow’s closing price** for a stock (e.g., AAL) will be **higher (Up)** or **lower (Down)** than today’s — i.e., a **binary classification** problem.

---

### 🧩 **Step-by-Step Conceptual Flow**

#### **1. Data Loading and Filtering**

* The dataset `SCOA_A5.csv` contains multiple stocks’ daily data — Open, High, Low, Close, Volume, etc.
* You choose one stock using:

  ```python
  STOCK_TICKER = 'AAL'
  df = data[data['Name'] == STOCK_TICKER]
  ```

  This isolates only **AAL’s** records so the model focuses on a single stock pattern.

#### **2. Sorting by Date**

* Stock data must be in chronological order for time-series learning:

  ```python
  df['date'] = pd.to_datetime(df['date'])
  df = df.sort_values('date').reset_index(drop=True)
  ```

#### **3. Target Creation (Supervised Label)**

* You define a binary label (`Target`) based on whether the next day’s close price is higher:

  ```python
  df['Target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
  ```

  * `1` = Price goes **UP** the next day
  * `0` = Price goes **DOWN**
* The last row has no "next day", so it's dropped with `dropna()`.

#### **4. Feature Selection and Scaling**

* Features chosen: `['open', 'high', 'low', 'close', 'volume']`.
* **MinMaxScaler** rescales all features to [0,1]:

  ```python
  scaler = MinMaxScaler()
  X = scaler.fit_transform(features)
  ```

  This helps the neural network train efficiently since large numeric ranges can bias weights.

---

#### **5. Train-Test Split**

* The dataset is divided as:

  ```python
  X_train, X_test, y_train, y_test = train_test_split(..., shuffle=False)
  ```

  * `shuffle=False` keeps the **temporal sequence** (important for time-series tasks).
  * Typically, 80% of early data for training, 20% later data for testing.

---

#### **6. ANN Model Construction**

You build a **feedforward neural network** (a simple ANN) using Keras:

```python
model = Sequential()
model.add(Dense(64, input_dim=5, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

**Layer Explanation:**

| Layer          | Neurons | Activation | Purpose                                 |
| -------------- | ------- | ---------- | --------------------------------------- |
| Input Layer    | 5       | —          | Receives open, high, low, close, volume |
| Hidden Layer 1 | 64      | ReLU       | Learns nonlinear relationships          |
| Hidden Layer 2 | 32      | ReLU       | Deepens representation                  |
| Output Layer   | 1       | Sigmoid    | Outputs probability of "UP" (0–1)       |

**Activations:**

* **ReLU (Rectified Linear Unit):** ( f(x) = \max(0, x) ) — efficient and avoids vanishing gradients.
* **Sigmoid:** Maps value to [0,1] — interpreted as probability of the next day being an “UP” day.

---

#### **7. Model Compilation**

```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

* **Loss Function:** Binary cross-entropy — suitable for binary classification.
* **Optimizer:** Adam — adaptive gradient descent optimizer.
* **Metric:** Accuracy — percentage of correct up/down predictions.

---

#### **8. Model Training**

```python
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)
```

* The network “learns” by adjusting internal weights for 100 epochs.
* Each epoch means one full pass over the training data.
* `batch_size=32` means it updates weights every 32 samples.

During training, it minimizes **binary cross-entropy loss**:
[
L = -\frac{1}{N}\sum [y \log(p) + (1-y)\log(1-p)]
]
where:

* ( y ) = actual label (0 or 1)
* ( p ) = predicted probability from sigmoid output

---

#### **9. Model Evaluation**

```python
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype("int32")
```

* The model outputs probabilities (e.g., 0.73 → likely UP).
* Anything > 0.5 → classified as 1 (UP), otherwise 0 (DOWN).

Then:

```python
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
```

---

#### **10. Confusion Matrix Interpretation**

Your confusion matrix:

```
[[117,   6],
 [118,  11]]
```

| Term         | Meaning                  | Count                           |
| ------------ | ------------------------ | ------------------------------- |
| **TN (117)** | Correct DOWN predictions | Model said “down” and was right |
| **FP (6)**   | Wrong UP predictions     | Said “up” but actually “down”   |
| **FN (118)** | Missed UP predictions    | Said “down” but actually “up”   |
| **TP (11)**  | Correct UP predictions   | Said “up” and was right         |

✅ The model correctly captures most **DOWN** trends,
❌ but misses many **UP** moves — hence low accuracy.

---

## ⚙️ Code Explanation (Line-by-Line Summary)

| Code Section                                                         | Purpose                             |
| -------------------------------------------------------------------- | ----------------------------------- |
| `import ...`                                                         | Import essential ML libraries       |
| `FILE_PATH = 'SCOA_A5.csv'`                                          | Define dataset location             |
| `STOCK_TICKER = 'AAL'`                                               | Select stock to analyze             |
| `data = pd.read_csv(FILE_PATH)`                                      | Load dataset into DataFrame         |
| `df = data[data['Name'] == STOCK_TICKER]`                            | Filter for selected stock           |
| `df['Target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)` | Create label column                 |
| `MinMaxScaler()`                                                     | Scale all features to [0,1]         |
| `train_test_split(..., shuffle=False)`                               | Maintain time order during split    |
| `Sequential()`                                                       | Build ANN model layer-by-layer      |
| `Dense(..., activation='relu')`                                      | Hidden layers for feature learning  |
| `Dense(..., activation='sigmoid')`                                   | Output layer for probability        |
| `compile(loss='binary_crossentropy', optimizer='adam')`              | Set training parameters             |
| `fit(..., epochs=100)`                                               | Train the network                   |
| `predict()`                                                          | Generate predictions on unseen data |
| `confusion_matrix()`                                                 | Measure model performance           |

---

## 📉 Conceptual Summary — Why Accuracy is ~50%

| Reason                 | Explanation                                                   |
| ---------------------- | ------------------------------------------------------------- |
| **Noisy data**         | Daily price fluctuations are almost random.                   |
| **Simple features**    | Only OHLCV; lacks momentum, trends, or volatility indicators. |
| **Static model**       | ANN doesn’t remember past trends (unlike LSTM).               |
| **Unbalanced targets** | More “down” days → model learns bias.                         |

---

## 🚀 Next Step Suggestions

1. **Add Technical Indicators** (RSI, EMA, SMA, Returns) — gives better predictive context.
2. **Use LSTM (Recurrent Network)** — captures time-dependence.
3. **Balance the Data** (using SMOTE or undersampling).
4. **Optimize Hyperparameters** — neurons, layers, learning rate.

---

"""