import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Download historical data
ticker = "AAPL"
data = yf.download(ticker, start="2015-01-01", end="2024-12-31")

# Step 2: Prepare the dataset
data = data[['Close']]
data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data['Date_Ordinal'] = data['Date'].map(pd.Timestamp.toordinal)

# Step 3: Train-Test Split
X = data[['Date_Ordinal']]
y = data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
predictions = model.predict(X_test)

# Step 6: Visualize results
plt.figure(figsize=(10, 5))
plt.scatter(data['Date'], y, color='blue', label="Actual Prices")
plt.plot(data['Date'], model.predict(X), color='red', label="Regression Line")
plt.title(f"{ticker} Stock Price Prediction using Linear Regression")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 7: Evaluation
print("Mean Squared Error (MSE):", mean_squared_error(y_test, predictions))
print("R² Score:", r2_score(y_test, predictions))
