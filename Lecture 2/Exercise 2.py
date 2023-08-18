import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Exercise 1:
print("Exercise 1")
# Method one
df = pd.read_csv('auto.csv', sep=',', na_values='?')
df = df.dropna()
# Method two
df2 = pd.read_csv('auto.csv')
df2 = df2[df2.horsepower != '?']

# Exercise 2:
print("Exercise 2")
# Don't plot against mpg or name
for i, name in enumerate(list(df.columns)):
  if name in ['mpg', 'name']:
    continue
  plt.subplot(2, 4, i)
  plt.plot(df[name], df['mpg'], '.', color='r', markersize=2)
  plt.ylabel('mpg')
  plt.xlabel(name)
plt.subplots_adjust(hspace=0.3, wspace=0.8)
plt.savefig("graph.png", dpi=200)
plt.show()



# Exercise 3:
print("Exercise 3")
# Plot the points
plt.plot(df['horsepower'], df['mpg'],'ro')
plt.ylabel('mpg'); plt.xlabel('horsepower')
plt.show()

# Fit the model
model = ols('mpg ~ horsepower', data=df).fit()
print(model.summary())

# Plot the line
horsepower_range = np.arange(min(df['horsepower']), max(df['horsepower']), 1)
# horsepower_range = np.linspace(min(df['horsepower']), max(df['horsepower']), len(df['horsepower']))
hp_values = model.params.Intercept + model.params.horsepower * horsepower_range

plt.plot(df['horsepower'], df['mpg'],'ro')
plt.plot(horsepower_range, hp_values)
plt.ylabel('mpg'); plt.xlabel('horsepower'); plt.title(f'mpg regressed on Horsepower')
plt.savefig("horsepower.png", dpi=200)
plt.show()



# Exercise 4:
print("Exercise 4")
# Since origin is a catagorical feature I will not include it in the model for the sake of simplicity. 
# You could add each catagori as a "dummy" feature if you wanted to.
model = ols(f'mpg ~ {" + ".join([f for f in list(df.columns) if f not in ["mpg", "name", "origin"]])}', data=df).fit()
print(model.summary())
# Acceleration, horsepower, cylinders and displacement are not relevant for the model
model = ols(f'mpg ~ {" + ".join([f for f in list(df.columns) if f not in ["mpg", "name", "origin", "acceleration", "horsepower", "cylinders", "displacement"]])}', data=df).fit()
print(model.summary())



# Exercise 5:
# In this exercise I chose to use the transformations on the whole dataset. 
# It would be a better idea to only use the appropriate transformations on the appropriate features.
# Example
# log(X) - Use when the relationship between Y and X seems to be exponential.
# sqrt(X) - Useful when there is a strong positive skewness in the data.
# X^2 - Use when there is a curvilinear relationship between Y and X.
# 1/x - Appropriate when there is a strong negative skewness in the data or a nonlinear relationship with a negative slope.

print("Exercise 5")
# Make it into a np.array for easy transformations!
df_array = np.array(df.iloc[:, : -1])

print('log(X)')
A = np.log(df_array)
df2 = pd.DataFrame(data=A, columns=list(df.columns)[:-1])

model = ols(f'mpg ~ {" + ".join([f for f in list(df2.columns) if f not in ["mpg", "origin"]])}', data=df2).fit()
print(model.summary())
# R^2 = 0.889
# cylinder and displacement have high p-values



print('sqrt(X)')
A = np.sqrt(df_array)
df2 = pd.DataFrame(data=A, columns=list(df.columns)[:-1])

model = ols(f'mpg ~ {" + ".join([f for f in list(df2.columns) if f not in ["mpg", "origin"]])}', data=df2).fit()
print(model.summary())
# R^2 = 0.861
# cylinders, displacement and acceleration have high p-values



print('X^2')
A = np.square(df_array)
df2 = pd.DataFrame(data=A, columns=list(df.columns)[:-1])

model = ols(f'mpg ~ {" + ".join([f for f in list(df2.columns) if f not in ["mpg", "origin"]])}', data=df2).fit()
print(model.summary())
# R^2 = 0.677
# all p-values low except horsepower which is 0.247



print('1/X')
A = np.reciprocal(df_array)
df2 = pd.DataFrame(data=A, columns=list(df.columns)[:-1])

model = ols(f'mpg ~ {" + ".join([f for f in list(df2.columns) if f not in ["mpg", "origin"]])}', data=df2).fit()
print(model.summary())
# R^2 = 0.851
# all p-values low except acceleration which is 0.757
