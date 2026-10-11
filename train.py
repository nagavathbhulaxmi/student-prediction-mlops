
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Create sample student dataset
data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 5, 7, 9],
    "attendance": [50, 55, 60, 65, 70, 75, 80, 90, 58, 72, 85, 95],
    "previous_marks": [30, 35, 40, 45, 50, 55, 65, 75, 38, 52, 70, 85],
    "final_marks": [25, 30, 38, 42, 50, 58, 68, 82, 32, 53, 73, 90]
}

df = pd.DataFrame(data)

# Prepare input and target
X = df[["study_hours", "attendance", "previous_marks"]]
y = df["final_marks"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print("Model training completed!")
print("Mean Absolute Error:", round(mae, 2))

# Save model
with open("student_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")
