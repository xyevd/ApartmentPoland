import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib

train_X = pd.read_csv("../data/processed/train_X.csv")
val_X = pd.read_csv("../data/processed/val_X.csv")
train_y = pd.read_csv("../data/processed/train_y.csv")
val_y = pd.read_csv("../data/processed/val_y.csv")

model = RandomForestRegressor(random_state=42)
# we have pd.dataframe, but we need pd.series
train_y = train_y.squeeze()
val_y = val_y.squeeze()

model.fit(train_X, train_y)
pred = model.predict(val_X)

rmse = root_mean_squared_error(val_y, pred)
percent_error = mean_absolute_percentage_error(val_y, pred) * 100
r2 = r2_score(val_y, pred)
print(f"Root mean squared error: {rmse:.2f}")
print(f"Mean absolute percentage error: {percent_error:.2f} %")
print(f"{r2}")

joblib.dump(model, '../models/random_forest_model.pkl')