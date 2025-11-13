import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense

# --- Configuration ---
FILE_PATH = 'SCOA_A5.csv'
STOCK_TICKER = 'AAL'

# Step 1: Load and filter data
try:
    data = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}")
    exit()

df = data[data['Name'] == STOCK_TICKER].copy()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Create target variable (1 = next day up, 0 = down)
df['Target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
df = df.dropna()

# Use only first 30,000 datapoints for faster training
df = df.head(30000)
print(f"Data loaded for {STOCK_TICKER}. Total records used: {len(df)}")

# Step 2: Feature selection and scaling
features = df[['open', 'high', 'low', 'close', 'volume']]
scaler = MinMaxScaler()
X = scaler.fit_transform(features)
y = df['Target'].values.astype(np.int32)

# Step 3: Train-test split (no shuffle for time series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4: Build and train ANN
model = Sequential([
    Dense(64, input_dim=5, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("\nTraining model...")
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
print("Training complete.")

# Step 5: Evaluate model
y_pred = (model.predict(X_test, verbose=0) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"\n--- Model Evaluation for {STOCK_TICKER} ---")
print(f"Test Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Step 6: User input for recommendation
print("\n--- Stock Investment Recommendation ---")
input_date = input("Enter a date (YYYY-MM-DD): ")
input_volume = float(input("Enter the trading volume for that date: "))

# Check if date exists in dataset
if input_date not in df['date'].astype(str).values:
    print("⚠️ Date not found in dataset. Please enter a valid date.")
else:
    # Get row for that date
    row = df[df['date'].astype(str) == input_date][['open', 'high', 'low', 'close']].iloc[0]
    input_data = np.array([[row['open'], row['high'], row['low'], row['close'], input_volume]])
    
    # Scale input data
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled, verbose=0)[0][0]
    
    print("\n📊 --- Prediction Result ---")
    print(f"Stock Name: {STOCK_TICKER}")
    print(f"Date: {input_date}")
    print(f"Entered Volume: {input_volume:,.0f}")
    
    if prediction > 0.5:
        print(f"✅ Recommendation: GOOD TO INVEST (Predicted Uptrend Probability: {prediction:.2f})")
    else:
        print(f"❌ Recommendation: NOT RECOMMENDED (Predicted Downtrend Probability: {prediction:.2f})")
