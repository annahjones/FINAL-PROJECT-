import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# === Step 1: Create training data ===
# 0 = unsafe, 1 = safe
X = [
    [7.0, 120.0, 500.0, 3.0, 250.0, 400.0, 10.0, 60.0, 3.0],  # safe
    [6.0, 100.0, 800.0, 5.0, 300.0, 450.0, 15.0, 90.0, 4.0],  # unsafe
    [7.2, 130.0, 400.0, 2.5, 230.0, 390.0, 8.0, 55.0, 2.5],   # safe
    [5.8, 80.0, 1500.0, 6.0, 100.0, 700.0, 20.0, 110.0, 5.0], # unsafe
    [7.1, 110.0, 300.0, 3.2, 240.0, 410.0, 9.0, 65.0, 3.2],   # safe
    [6.2, 90.0, 1000.0, 4.0, 150.0, 600.0, 18.0, 100.0, 4.5]  # unsafe
]
y = [1, 0, 1, 0, 1, 0]

# === Step 2: Train the model ===
model = RandomForestClassifier()
model.fit(X, y)

# === Step 3: Save the model ===
model_bytes = pickle.dumps(model)
print("Model exported successfully. Bytes:")
print(model_bytes[:100])  # Preview first 100 bytes
