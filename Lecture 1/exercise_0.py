# 1. import NumPy in a python script
import numpy as np

a = np.full((2, 3), 4)
b = np.array([[1, 2, 3], [4, 5, 6]])
c = np.eye(2, 3)
d = a + b + c

a = np.array([[1, 2, 3, 4, 5],
              [5, 4, 3, 2, 1],
              [6, 7, 8, 9, 0], 
              [0, 9, 8, 7, 6]])

df = len(a)
# df = np.sum(a)
print(df)

np_transpose = a.transpose()
print(np_transpose)

# 2. Pandas 
# import pandas as pd

# autos = pd.read_csv("auto.csv")
# print(autos.head())

# if autos['mpg'] >= 16:
#     print(autos.head())

# # autos = autos[autos['mpg'] >= 16]
# # print(autos.head())

# print(autos[0:7][["weight", "horsepower"]])

# if autos['horsepower'] != '?':
#     print(int(autos['horsepower']))
    
# df2 = autos.mean()

# 3. Matplotlib

import matplotlib.pyplot as plt

a = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34])
b = np.array([1, 8, 28, 56, 70, 56, 28, 8, 1])

epochs = a
accuracy = b

epochs = range(1, 10)

plt.plot(epochs, a, label = "training accuracy", linestyle ="-")
plt.plot(epochs, b, label = "validation accuracy", linestyle ="--")
plt.title("Training and validation accuracy")
plt.legend()
plt.show()

# 4. PyTorch
import torch 

matrix_a = torch.rand((3, 3))
matrix_b = torch.rand((3, 3))

print(matrix_a)
print(matrix_b)

result = torch.matmul(matrix_a, matrix_b)

print(result)