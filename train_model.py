import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Sample dataset with 9 features and target 'Potability' (1 = safe, 0 = unsafe)
data = {
    "ph": [7.0, 3.5, 8.1, 6.8, 5.5],
    "Hardness": [200, 300, 150, 180, 400],
    "Solids": [10000, 15000, 12000, 18000, 20000],
    "Chloramines": [6.5, 4.5, 9.0, 7.2, 3.0],
    "Sulfate": [250, 300, 200, 150, 100],
    "Conductivity": [400, 500, 450, 480, 550],
    "Organic_carbon": [10, 12, 15, 9, 20],
    "Trihalomethanes": [75, 90, 60, 85, 100],
    "Turbidity": [2.0, 4.0, 3.5, 1.5, 5.0],
    "Potability": [1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

# Features and target
X = df.drop("Potability", axis=1)
y = df["Potability"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model to a file
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ model.pkl has been created.")
