import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("/workspaces/mlops/src/data/processed/test_processed_data.csv")
X_train = train_data.iloc[:, :-1].values  # Features
y_train = train_data.iloc[:, -1].values   # Target variable

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

pickle.dump(clf, open("model.pkl", "wb"))
