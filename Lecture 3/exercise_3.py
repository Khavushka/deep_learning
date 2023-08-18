'''
- Load the auto.csv dataset again using the pandas.read function and remember to remove the missing values in the dataset, indicated by "?", and then make sure the corresponding columns are casted to a numerical type. 
'''

import pandas as pd

df = pd.read_csv('Lecture 3/auto.csv', sep=',', na_values='?')

# df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)

numeric_columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
print(df.head())

'''
- Inspect the data. Plot the relationships between the different variables and mpg. Use for example the matplotlib.pyplot scatter plot. Do you already suspect what features might be helpful to regress the consumption? Save the praph
'''
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(df['displacement'], df['mpg'], label='Displacement')
plt.scatter(df['horsepower'], df['mpg'], label='Horsepower')
plt.scatter(df['weight'], df['mpg'], label='Weight')
plt.scatter(df['acceleration'], df['mpg'], label='Acceleration')
plt.xlabel('Feature Values')
plt.ylabel('MPG')
plt.title('Relationships between Features and MPG')
plt.legend()
plt.grid()

# Save the plot
plt.savefig('mpg_relationships.png')

plt.show()

'''
- Perform a linear regression using the OLS function from the statsmodels package. Use the following 'horsepower' as feature and regress the value 'mpg'. It is a good idea to look up the statsmodel documentation on OLS, to understand how to use it. Further, plot the result including your regression line.
'''
import statsmodels.api as sm

X = df['horsepower'] # Independent variable
y = df['mpg'] # Dependent variable

X = sm.add_constant(X)

# Linear regression using OLS
model = sm.OLS(y, X).fit()
print(model.summary())

plt.figure(figsize=(10, 6))
plt.scatter(df['horsepower'], df['mpg'], label='Data')
plt.plot(X['horsepower'], model.predict(X), color='red', label='Regression Line')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.title('Linear Regression: Horsepower vs. MPG')
plt.legend()
plt.grid()
plt.show()

'''
- Now extend the model using all features. How would you determine which features are important and which aren't? Try to find a good selection of features for your model.
'''


# Define the independent variables (features) and the dependent variable
X = df.drop(['mpg'], axis=1)  # All features except 'mpg'
y = df['mpg']

# Add a constant term to the independent variables (intercept)
X = sm.add_constant(X)

# Perform linear regression using OLS
model = sm.OLS(y, X).fit()

# Print the summary of the regression
print(model.summary())

# Extract feature importance scores (absolute values of coefficients)
feature_importance = pd.Series(abs(model.params.values[1:]), index=X.columns[1:])

# Sort the feature importance scores in descending order
sorted_feature_importance = feature_importance.sort_values(ascending=False)

# Plot the feature importance scores
plt.figure(figsize=(10, 6))
sorted_feature_importance.plot(kind='bar')
plt.xlabel('Features')
plt.ylabel('Absolute Coefficient Values')
plt.title('Feature Importance Scores')
plt.xticks(rotation=45)
plt.grid()

# Show the plot
plt.show()

# # Select a subset of features based on importance
# selected_features = sorted_feature_importance[sorted_feature_importance > threshold].index.tolist()
# print("Selected features:", selected_features)


'''
- Can you improve your regression performance by trying differnet transformations of the variables, such as  𝑙og(𝑋),√𝑋,1/𝑋,𝑋^2, and so on. For each transformation, which features are important and which aren't?

* Better understanding the data
* Better understanding a model
* Reducing the number of input features
'''
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import statsmodels.api as sm

# # Load the dataset
# df = pd.read_csv('auto.csv')

# # Replace '?' with NaN and drop rows with missing values
# df.replace('?', pd.NA, inplace=True)
# df.dropna(inplace=True)

# # Cast columns to numerical type
# numeric_columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']
# df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

# # Define the independent variables (features) and the dependent variable
# X = df.drop(['mpg'], axis=1)  # All features except 'mpg'
# y = df['mpg']

# # Add a constant term to the independent variables (intercept)
# X = sm.add_constant(X)

# # List of transformations to try
# transformations = [np.log, np.sqrt, lambda x: 1/x, lambda x: x**2]

# # Perform linear regression with different transformations
# for transform in transformations:
#     transformed_X = X.applymap(transform)
#     model = sm.OLS(y, transformed_X).fit()
    
#     print(f"\nTransformation: {transform.__name__}")
#     print(model.summary())

#     # Extract feature importance scores (absolute values of coefficients)
#     feature_importance = pd.Series(abs(model.params.values[1:]), index=X.columns[1:])
    
#     # Sort the feature importance scores in descending order
#     sorted_feature_importance = feature_importance.sort_values(ascending=False)
    
#     # Plot the feature importance scores
#     plt.figure(figsize=(10, 6))
#     sorted_feature_importance.plot(kind='bar')
#     plt.xlabel('Features')
#     plt.ylabel('Absolute Coefficient Values')
#     plt.title(f'Feature Importance Scores - {transform.__name__}')
#     plt.xticks(rotation=45)
#     plt.grid()
    
#     # Show the plot
#     plt.show()

#     # Select a subset of features based on importance
#     threshold = 0.1  # Adjust the threshold as needed
#     selected_features = sorted_feature_importance[sorted_feature_importance > threshold].index.tolist()
#     print("Selected features:", selected_features)
