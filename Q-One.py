import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
output_file = "E:/Ml-Training/DataSet/housePrice.csv"
df = pd.read_csv(output_file)
# Show Five Record
print(df.head(5))
# Division Data To the Experiment And Test
X = df.drop(['Area'], axis=1)  #
X = pd.get_dummies(X, columns=["Address"])
y = df["Area"]  #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create Model LinearRegression
model = LinearRegression()
# Learn With Data
model.fit(X_train, y_train)
# Prediction With DataTest
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)  # Error
r2 = r2_score(y_test, y_pred)  # The coefficient of determination (R-squared)
# Print Result
print("Mean Squared Error:", mse)
print("R-squared:", r2)