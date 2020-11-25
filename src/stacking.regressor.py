# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 00:29:48 2020

@author: amosn
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 00:45:48 2020

@author: amosn
"""

# Import libraries
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
# 
from sklearn.model_selection import RepeatedKFold
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR

# Compare machine learning models for regression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import StackingRegressor

# Import additional libraries
from xgboost import XGBClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost as xgb
from datetime import datetime


# Read in the dataset as Pandas DataFrame
abalone = pd.read_csv(r'C:\Users\amosn\OneDrive\Desktop\datascience\Applied Machine Learning Ensemble Modeling\data\abalone.csv')

# Look at data using the info() function
abalone.info()

# Look at data using the describe() function
abalone.describe()

# Print the first 5 rows of the data using the head() function
abalone.head()

# Convert Pandas DataFrame to numpy array - Return only the values of the DataFrame with DataFrame.to_numpy()
abalone = abalone.to_numpy()

# Create X matrix and y (target) array using slicing [row_start:row_end, 1:target_col],[row_start:row_end, target_col] - Removing 1st column by starting at index 1
X, y = abalone[:, 1:-1], abalone[:, -1]

# Print X matrix and y (target) array dimensions using .shape
print('Shape: %s, %s' % (X.shape,y.shape))

# Convert y (target) array to 'float32' using .astype()
y = y.astype('float32')

# Creating a Naive Regressor

# Evaluate naive

# Instantiate a DummyRegressor with 'median' strategy
naive = DummyRegressor(strategy='median')

# Create RepeatedKFold cross-validator with 10 folds, 3 repeats and a seed of 1.
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# Calculate accuracy using `cross_val_score()` with model instantiated, data to fit, target variable, 'neg_mean_absolute_error' scoring, cross validator, n_jobs=-1, and error_score set to 'raise'
n_scores = cross_val_score(naive, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')

# Print mean and standard deviation of n_scores:
print('Baseline: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# Creating a Baseline Regressor

# Evaluate baseline model

# Instantiate a Support Vector Regressor with 'rbf' kernel, gamma set to 'scale', and regularization parameter set to 10
model = SVR(kernel='rbf',gamma='scale',C=10)

# Calculate accuracy using `cross_val_score()` with model instantiated, data to fit, target variable, 'neg_mean_absolute_error' scoring, cross validator 'cv', n_jobs=-1, and error_score set to 'raise'
m_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')

# Print mean and standard deviation of m_scores: 
print('Good: %.3f (%.3f)' % (mean(m_scores), std(m_scores)))

# Custom function # 1: get_stacking()

# Define get_stacking():
def get_stacking():

	# Create an empty list for the base models called layer1
  layer1 = list()

  # Append tuple with classifier name and instantiations (no arguments) for KNeighborsRegressor, DecisionTreeRegressor, and SVR base models
  # Hint: layer1.append(('ModelName', Classifier()))
  layer1.append(('KNN', KNeighborsRegressor()))
  layer1.append(('DT', DecisionTreeRegressor()))
  layer1.append(('SVM', SVR()))

  # Instantiate Linear Regression as meta learner model called layer2
  layer2 = LinearRegression()

	# Define Stackingregressor() called model passing layer1 model list and meta learner with 5 cross-validations
  model = StackingRegressor(estimators=layer1, final_estimator=layer2, cv=5)

  # return model
  return model


#Custom function # 2: get_models()
  # Define get_models():
def get_models():

  # Create empty dictionary called models
  models = dict()

  # Add key:value pairs to dictionary with key as ModelName and value as instantiations (no arguments) for KNeighborsRegressor, DecisionTreeRegressor, and SVR base models
  # Hint: models['ModelName'] = Classifier()
  models['KNN'] = KNeighborsRegressor()
  models['DT'] = DecisionTreeRegressor()
  models['SVM'] = SVR()

  # Add key:value pair to dictionary with key called Stacking and value that calls get_stacking() custom function
  models['Stacking'] = get_stacking()

  # return dictionary
  return models



# Custom function # 3: evaluate_model(model)
# Define evaluate_model:
def evaluate_model(model):

  # Create RepeatedKFold cross-validator with 10 folds, 3 repeats and a seed of 1.
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
 
  # Calculate accuracy using `cross_val_score()` with model instantiated, data to fit, target variable, 'neg_mean_absolute_error' scoring, cross validator 'cv', n_jobs=-1, and error_score set to 'raise'
	scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
 
  # return scores
	return scores

# Assign get_models() to a variable called models
models = get_models()


# Evaluate the models and store results
# Create an empty list for the results
results = list()

# Create an empty list for the model names
names = list()

# Create a for loop that iterates over each name, model in models dictionary 
for name, model in models.items():

	# Call evaluate_model(model) and assign it to variable called scores
	scores = evaluate_model(model)
 
  # Append output from scores to the results list
	results.append(scores)
 
  # Append name to the names list
	names.append(name)
 
  # Print name, mean and standard deviation of scores:
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
 
# Plot model performance for comparison using names for x and results for y and setting showmeans to True
sns.boxplot(x=names, y=results, showmeans=True)


# Double Stacking-2 Layers

# Define get_stacking() - adding another layer:
def get_stacking():

	# Create an empty list for the 1st layer of base models called layer1
  layer1 = list()

  # Create an empty list for the 2nd layer of base models called layer2
  layer2 = list()

  # Append tuple with classifier name and instantiations (no arguments) for KNeighborsRegressor, DecisionTreeRegressor, and SVR base models
  # Hint: layer1.append(('ModelName', Classifier()))
  layer1.append(('KNN', KNeighborsRegressor()))
  layer1.append(('DT', DecisionTreeRegressor()))
  layer1.append(('SVM', SVR()))

  # Append tuple with classifier name and instantiations (no arguments) for KNeighborsRegressor, DecisionTreeRegressor, and SVR base models
  # Hint: layer2.append(('ModelName', Classifier()))
  layer2.append(('KNN', KNeighborsRegressor()))
  layer2.append(('DT', DecisionTreeRegressor()))
  layer2.append(('SVM', SVR()))

	# Define meta learner StackingRegressor() called layer3 passing layer2 model list to estimators, LinearRegression() to final_estimator with 5 cross-validations
  layer3 = StackingRegressor(estimators=layer2, final_estimator=LinearRegression(), cv=5)

	# Define Stackingregressor()  called model passing layer1 model list to estimators and meta learner (layer3) to final_estimator with 5 cross-validations
  model = StackingRegressor(estimators=layer1, final_estimator=layer3, cv=5)

  # return model
  return model

# Assign get_models() to a variable called models
models = get_models()

# Evaluate the models and store results
# Create an empty list for the results
results = list()

# Create an empty list for the model names
names = list()

# Create a for loop that iterates over each name, model in models dictionary 
for name, model in models.items():

	# Call evaluate_model(model) and assign it to variable called scores
	scores = evaluate_model(model)
 
  # Append output from scores to the results list
	results.append(scores)
 
  # Append name to the names list
	names.append(name)
 
  # Print name, mean and standard deviation of scores:
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
 
# Plot model performance for comparison using names for x and results for y and setting showmeans to True
sns.boxplot(x=names, y=results, showmeans=True)