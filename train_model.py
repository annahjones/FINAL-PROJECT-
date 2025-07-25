import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Simulate sample dataset with 9 features
np.random.seed(42)
X = np.random.rand(1000, 9) * 100  # 1000 samples, 9 features
y = (X[:, 0] + X[:, 1] * 0.5 > 70).astype(int)  # Just a dummy logic for label

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model to a file
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ model.pkl created successfully.")
