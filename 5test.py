import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense

# Step 1: Load dataset from CSV
# Replace 'your_dataset.csv' with your actual file name/path
df = pd.read_csv('SCOA_A5.csv')

# Step 2: Create Target column (1 if next day's Close > today's Close, else 0)
df['Target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

# Step 3: Select features (ignoring 'Name' column)
features = df[['open', 'high', 'low', 'close', 'volume']]

# Step 4: Normalize features
scaler = MinMaxScaler()
X = scaler.fit_transform(features)
y = df['Target'].values

# Step 5: Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 6: Build ANN model
model = Sequential()
model.add(Dense(64, input_dim=5, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Step 7: Evaluate model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)