import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("Housing.csv")
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=["price"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

r2   = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"R² Score : {r2:.2f}  (1.0 = perfect)")
print(f"RMSE     : {rmse:,.0f}")

print("\nSample Predictions:")
print(f"{'Actual':>12}  {'Predicted':>12}")
for actual, pred in zip(y_test.values[:5], predictions[:5]):
    print(f"{actual:>12,.0f}  {pred:>12,.0f}")