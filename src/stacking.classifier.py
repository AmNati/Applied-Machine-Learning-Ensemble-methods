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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier


# Import several other classifiers for ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier


# Import additional libraries
from xgboost import XGBClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost as xgb
from datetime import datetime


# Read in the dataset as Pandas DataFrame
diabetes = pd.read_csv(r'C:\Users\amosn\OneDrive\Desktop\datascience\Applied Machine Learning Ensemble Modeling\data\pima_indians_diabetes.csv')

# Look at data using the info() function-
#Critical for preliminary understanding of your data
diabetes.info()

# Look at data using the describe() function-Gives the summary statistics of the data
diabetes.describe()

# Print the first 5 rows of the data using the head() function
diabetes.head()

# Convert Pandas DataFrame to numpy array - Return only the values of the DataFrame with DataFrame.to_numpy()
diabetes = diabetes.to_numpy()


#Create X matrix and y (target) array using slicing [row_start:row_end, col_start:target_col],[row_start:row_end, target_col]
X, y = diabetes[:, :-1], diabetes[:, -1]
print('Shape: %s, %s' % (X.shape, y.shape))

# Convert X matrix data types to 'float32' for consistency using .astype()
X = X.astype('float32')

# Convert y (target) array to 'str' using .astype()
y = y.astype('str')

# Encode class labels in y array using dot notation with LabelEncoder().fit_transform()
# Hint: y goes in the fit_transform function call
y = LabelEncoder().fit_transform(y)


# Naive/Null Classifier

# Instantiate a DummyClassifier with 'most_frequent' strategy
naive = DummyClassifier(strategy='most_frequent')

# Create RepeatedStratifiedKFold cross-validator with 10 folds, 3 repeats and a seed of 1.
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# Calculate accuracy using `cross_val_score()` with model instantiated, data to fit, target variable, 'accuracy' scoring, cross validator, n_jobs=-1, and error_score set to 'raise'
n_scores = cross_val_score(naive, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# Print mean and standard deviation of n_scores: 
print('Naive score: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# Creating a Baseline Classifier
# Evaluate baseline model

# Instantiate a DecisionTreeClassifier
model = DecisionTreeClassifier()

# Calculate accuracy using `cross_val_score()` with model instantiated, data to fit, target variable, 'accuracy' scoring, cross validator 'cv', and error_score set to 'raise'
m_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# Print mean and standard deviation of m_scores: 
print('Baseline score: %.3f (%.3f)' % (mean(m_scores), std(m_scores)))


# Custom function # 1: get_stacking()

# Define get_stacking():
def get_stacking():

	# Create an empty list for the base models called layer1
  layer1 = list()

  # Append tuple with classifier name and instantiations (no arguments) for KNeighborsClassifier, SVC, and GaussianNB base models
  # Hint: layer1.append(('ModelName', Classifier()))
  layer1.append(('DT', DecisionTreeClassifier()))
  layer1.append(('KNN', KNeighborsClassifier()))
  layer1.append(('SVM', SVC()))
  layer1.append(('Bayes', GaussianNB()))

  # Instantiate Logistic Regression as meta learner model called layer2
  layer2 = LogisticRegression()

	# Define StackingClassifier() called model passing layer1 model list and meta learner with 5 cross-validations
  model = StackingClassifier(estimators=layer1, final_estimator=layer2, cv=5)

  # return model
  return model

# Custom function # 2: get_models()

# Define get_models():
def get_models():

  # Create empty dictionary called models
  models = dict()

  # Add key:value pairs to dictionary with key as ModelName and value as instantiations (no arguments) for KNeighborsClassifier, SVC, and GaussianNB base models
  # Hint: models['ModelName'] = Classifier()
  models['DT'] = DecisionTreeClassifier() 
  models['KNN'] = KNeighborsClassifier() 
  models['SVM'] = SVC()
  models['Bayes'] = GaussianNB()

  # Add key:value pair to dictionary with key called Stacking and value that calls get_stacking() custom function
  models['Stacking'] = get_stacking()

  # return dictionary
  return models

# Custom function # 3: evaluate_model(model)
# Define evaluate_model:
def evaluate_model(model):

  # Create RepeatedStratifiedKFold cross-validator with 10 folds, 3 repeats and a seed of 42.
  cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

  # Calculate accuracy using `cross_val_score()` with model instantiated, data to fit, target variable, 'accuracy' scoring, cross validator 'cv', n_jobs=-1, and error_score set to 'raise'
  scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

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


#Custom function # 4: best_model(name, model)
# Define best_model:
def best_model(name, model):
  pipe = Pipeline([('scaler', StandardScaler()), ('classifier',model)])  

  if name == 'SVM':
    param_grid = {'classifier__kernel' : ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']} 
    # Create grid search object
    # this uses k-fold cv
    clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, n_jobs=-1)

    # Fit on data
    best_clf = clf.fit(X, y)

    best_hyperparams = best_clf.best_estimator_.get_params()['classifier']

    return name, best_hyperparams 

  if name == 'Bayes': 
    param_grid = {'classifier__var_smoothing' : np.array([1e-09, 1e-08])} 
    # Create grid search object
    # this uses k-fold cv

    clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, n_jobs=-1)

    # Fit on data
    best_clf = clf.fit(X, y)

    best_hyperparams = best_clf.best_estimator_.get_params()['classifier']

    return name, best_hyperparams 

  if name == 'RF': 
    param_grid = {'classifier__criterion' : np.array(['gini', 'entropy']),
                  'classifier__max_depth' : np.arange(5,11)} 
    # Create grid search object
    # this uses k-fold cv

    clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, n_jobs=-1)

    # Fit on data
    best_clf = clf.fit(X, y)

    best_hyperparams = best_clf.best_estimator_.get_params()['classifier']
 
    return name, best_hyperparams  

  if name == 'XGB':
    param_grid = {'classifier__learning_rate' : np.arange(0.022,0.04,.01),
                  'classifier__max_depth' : np.arange(5,10)} 
    # Create grid search object
    # this uses k-fold cv
    clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5,  n_jobs=-1)

    # Fit on data
    best_clf = clf.fit(X, y)
    best_hyperparams = best_clf.best_estimator_.get_params()['classifier']

    return name, best_hyperparams  

##  Adding Random Forest and XGBoost to our get_stacking() custom function in layer 1 (and removing the poorest performers DT and KNN
    
def get_stacking():

	# Create an empty list for the base models called layer1
  layer1 = list()

  # Append tuple with classifier name and instantiations (no arguments) for SVC and GaussianNB base models AND call cust fx #4 best_model on each
  # Hint: layer1.append((best_model('ModelName', Classifier())))
  layer1.append((best_model('SVM', SVC())))
  layer1.append((best_model('Bayes', GaussianNB())))

  # Add RandomForestClassifier and xgb.XGBClassifier as base models
  layer1.append((best_model('RF', RandomForestClassifier())))
  layer1.append((best_model('XGB', xgb.XGBClassifier())))

  # Instantiate Logistic Regression as meta learner model called layer2
  layer2 = LogisticRegression()

	# Define StackingClassifier() called model passing layer1 model list and meta learner with 5 cross-validations
  model = StackingClassifier(estimators=layer1, final_estimator=layer2, cv=5)

  # return model
  return model

# Adding Random Forest and XGBoost to our get_models() custom function:
# Define get_models():
def get_models():

  # Create empty dictionary called models
  models = dict()

  # Add key:value pairs to dictionary with key as ModelName and value as instantiations (no arguments) for SVC and GaussianNB base models
  # Hint: models['ModelName'] = Classifier() 
  models['SVM'] = SVC()
  models['Bayes'] = GaussianNB()

  # we'll add two more classifers to the mix - RandomForestClassifier and xgb.XGBClassifier
  models['RF'] = RandomForestClassifier()
  models['XGB'] = xgb.XGBClassifier()


  # Add key:value pair to dictionary with key called Stacking and value that calls get_stacking() custom function
  models['Stacking'] = get_stacking()

  # return dictionary
  return models

# Assign get_models() to a variable called models
models = get_models()


#Custom function # 5: evaluate_model(model)
# Define evaluate_model(model):
def evaluate_model(model):

  # Create RepeatedStratifiedKFold cross-validator with 10 folds, 3 repeats and a seed of 1.
  cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

  # Calculate accuracy using `cross_val_score()` with model instantiated, data to fit, target variable, 'accuracy' scoring, cross validator 'cv', n_jobs=-1, and error_score set to 'raise'
  scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

  # return scores
  return scores

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