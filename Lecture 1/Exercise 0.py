# 1a Import NumPy in a python script
import numpy as np

# 1b Think about which values are in the numpy array in d, then verify that you were right
a = np.full((2, 3), 4)
b = np.array([[1, 2, 3], [4, 5, 6]])
c = np.eye(2, 3)
d = a + b + c

# 1c Get the third row from a as a rank 2 array
a = np.array([[1, 2, 3, 4, 5],
              [5, 4, 3, 2, 1],
              [6, 7, 8, 9, 0],
              [0, 9, 8, 7, 6]])
print(a[2:3, :])

# 1d Sum the rows of a
a = np.array([[1, 2, 3, 4, 5],
              [5, 4, 3, 2, 1],
              [6, 7, 8, 9, 0],
              [0, 9, 8, 7, 6]])
print(np.sum(a, axis=1))

# 1e Get the transpose of a
a = np.array([[1, 2, 3, 4, 5],
              [5, 4, 3, 2, 1],
              [6, 7, 8, 9, 0],
              [0, 9, 8, 7, 6]])
print(a.T)

# 2a Import pandas
import pandas as pd

# 2b Read the file auto.csv
autos = pd.read_csv("Lecture 1/auto.csv")
print(autos.head())

# 2c Remove all rows with mpg lower than 16
autos = autos[autos['mpg'] >= 16]
print(autos.head())

# 2d Get the first 7 rows of the columns weight and acceleration
print(autos[0:7][["weight", "horsepower"]])

# 2e Remove the rows in the horsepower column that has the value '?', and convert it to an int type instead of a string type
autos = autos[autos.horsepower != '?']
autos = autos.astype({'horsepower': 'int32'})

# 2f Calculate the averages of every column except name
length = autos.shape[0]
autos = autos.values[:, :8]
auto_sums = np.sum(autos, axis=0)
auto_avgs = auto_sums / length
print(auto_avgs)

# 3a Import matplotlib
import matplotlib.pyplot as plt

# 3b Make a plot with two lines from a and b. Name the first axis Epochs, and the other Accuracy.
# Call the line made from a Training accuracy and the line made from b Validation Accuracy.
# Give the plot the title "Training and validation accuracy" and Show the plot
a = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34])
b = np.array([1, 8, 28, 56, 70, 56, 28, 8, 1])

epochs = range(1, 10)

plt.plot(epochs, a, 'b', label='Training Accuracy')
plt.plot(epochs, b, 'g', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# 4a Import PyTorch
import torch

# 4b Create two random matrices using PyTorch’s (torch.rand) of size (3x3).
matrix_a = torch.rand((3, 3))
matrix_b = torch.rand((3, 3))

print(matrix_a)
print(matrix_b)

# 4c Multiply the two matrices using PyTorch's matrix multiplication function (torch.matmul).
result = torch.matmul(matrix_a, matrix_b)

print(result)
