import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix


import os.path

def cross_val_polnomial_reg(X, y, degrees, cv):
    '''
    Transforms X into polynomial features.
    Calculates the average MSE for selected number of cross-validations.
    Returns an array for each selectd number of degrees.
    
    '''
    mse_arr = []
    
    for degree in degrees:
        X_poly = PolynomialFeatures(degree=degree).fit_transform(X)
        linear_model = LinearRegression()

        c_val = cross_val_score(linear_model, X=X_poly, y=y, cv=10, scoring='neg_mean_squared_error', n_jobs=-1).mean()
        mse_arr.append(-c_val)
        
    return mse_arr

def plot_mse_per_polynomial_degree(degrees, mse_arr):
    '''
    Produces line plot showing the MSE for each degree of polynomial
    '''
    fig, ax = plt.subplots(figsize=(7,4))

    ax.plot(degrees, mse_arr, c='b', linewidth=2, marker='*')
    ax.set_xlabel('degrees')
    ax.set_ylabel('MSE')
    ax.set_title('Polynomial Regression', fontsize=16)
    fig.tight_layout();
    

def calc_and_plot_mse_for_polynomial_degrees(X, y, degrees, cv):
    '''
    Coasleces the cross-validation MSE calculation and line plot functions.
    
    '''
    mse_arr = cross_val_polnomial_reg(X, y, degrees, cv)
    plot_mse_per_polynomial_degree(degrees, mse_arr)