import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ✅ Step 1: More data added
data = pd.DataFrame({
    'SquareFootage': [1500, 2000, 1200, 1800, 2200, 2500, 1400, 1900, 1600, 1700],
    'Bedrooms': [3, 4, 2, 3, 4, 5, 2, 3, 3, 3],
    'Bathrooms': [2, 3, 1, 2, 3, 4, 1, 2, 2, 2],
    'Price': [300000, 400000, 200000, 350000, 450000, 500000, 220000, 360000, 310000, 330000]
})

X = data[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Predicted Prices:", predictions)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, predictions))
print("R² Score:", r2_score(y_test, predictions))

# ✅ Feature name fix in prediction
new_house = pd.DataFrame([[1600, 3, 2]], columns=['SquareFootage', 'Bedrooms', 'Bathrooms'])
predicted_price = model.predict(new_house)
print("New House Price Prediction:", predicted_price[0])