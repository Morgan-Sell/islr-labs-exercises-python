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


def linear_reg_predictions(X_train, y_train, X_test):
    '''
    Fits a linear regression and calculates predictions
    '''
    lin_reg = LinearRegression().fit(X_train, y_train)
    pred  = lin_reg.predict(X_test)
    return pred


def calc_residuals(y_pred, y_true):
    '''
    Calculates residuals
    '''
    return y_pred - y_true


def calc_rss(y_pred, y_true):
    '''
    Calculates the residual sum of squares
    '''
    residuals = calc_residuals(y_pred, y_true)
    rss = np.sum(np.square(residuals))
    return rss

def plot_linear_reg_model_and_residuals(X_test, y_test, y_pred, residuals):
    '''
    Creates twos plots: (1) linear regression; and (2) residuals
    '''
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    # plot 1
    ax[0].scatter(X_test, y_test, c='g', alpha=0.3)
    ax[0].plot(X_test, y_pred, c='b', linewidth=2)
    ax[0].set_title('Linear Regression', fontsize=16)

    # plot 2
    ax[1].scatter(X_test, residuals, c='orange', alpha=0.8)
    ax[1].axhline(y=0)
    ax[1].set_title('Residuals', fontsize=16)

    fig.tight_layout();
    

def fit_lin_reg_to_create_plots_and_residuals(X_train, y_train, X_test, y_true):
    '''
    Coasceleces 3 formulas to fit linear regression, calculate residuals and plot the results.
    
    '''
    pred = linear_reg_predictions(X_train, y_train, X_test)
    residuals = calc_residuals(pred, y_true)
    plot_linear_reg_model_and_residuals(X_test, y_true, pred, residuals)

def transform_variables_to_polynomial(X_train, X_test, num_degrees):
    '''
    Transforms indepedent variables to polynomial features.
    '''
    poly = PolynomialFeatures(degree=num_degrees)
    poly.fit(X_train)
    X_train_poly = poly.transform(X_train)
    X_test_poly = poly.transform(X_test)

    return X_train_poly, X_test_poly

def plot_poly_reg_model_and_residuals(X, y, residuals, num_degrees):
    '''
    Creates twos plots: (1) linear regression; and (2) residuals
    '''
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    # Scatter plot w/ polynomial regression
    ax[0].scatter(X, y, c='g', alpha=0.3)
    sns.regplot(X, y, order=num_degrees, truncate=True, scatter=False, ax=ax[0])
    ax[0].set_title('Polynomial Regression', fontsize=16)

    # residual plot
    ax[1].scatter(X, residuals, c='orange', alpha=0.8)
    ax[1].axhline(y=0)
    ax[1].set_title('Residuals', fontsize=16)

    fig.tight_layout();

def create_fit_polynomial_reg_and_graph_predictions_and_residuals(X_train, y_train, X_test, y_true, num_degrees):
    '''
    Coaslesces multiple functions to transform features and fit/predict a polynomial regression.
    Visualizes results.
    '''
    X_train_poly, X_test_poly = transform_variables_to_polynomial(X_train, X_test, num_degrees)
    # Polynomial regression b/c X_train and X_test have been transformed.
    pred = linear_reg_predictions(X_train_poly, y_train, X_test_poly)
    residuals = calc_residuals(pred, y_true)
    plot_poly_reg_model_and_residuals(X_test, y_true, residuals, num_degrees)