from sklearn.datasets import fetch_openml
import numpy as np

# fetch MNIST — as_frame=False gives you plain numpy arrays, not a pandas DataFrame
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X, y = mnist["data"], mnist["target"]

print("X shape:", X.shape)      
print("y shape:", y.shape)
print("X dtype:", X.dtype)
print("y dtype:", y.dtype)      

print("First label:", y[0])
print("First sample min/max pixel value:", X[0].min(), X[0].max())

one_digit = X[0].reshape(28, 28)
print("Reshaped:", one_digit.shape)