import pandas as pd
import numpy as np
import json
import pickle
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score  

test_data = pd.read_csv("/workspaces/mlops/src/data/processed/test_processed_data.csv")

X_test = test_data.iloc[:, :-1].values  # Features
y_test = test_data.iloc[:, -1].values   # Target variable

model = pickle.load(open("model.pkl", "rb"))

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)           
f1 = f1_score(y_test, y_pred)

metrics_dict = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1  
}

with open("metrics.json", "w") as file:
    json.dump(metrics_dict, file, indent=4)