import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Load Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_data = pd.read_csv(url, names=names)

# Separate features (inputs) and labels (targets)
X = iris_data.iloc[:, :-1].values
T = iris_data.iloc[:, -1].values

# Encode target dataset to one-hot encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(T)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
T_onehot = onehot_encoder.fit_transform(integer_encoded)

print("Input (X):")
print(X[:5])  # Print first 5 rows of input data
print("\nTarget (T) - One-hot encoded:")
print(T_onehot[:5])  # Print first 5 rows of one-hot encoded target data
