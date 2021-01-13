import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pydot
from IPython.display import Image
from io import StringIO
import os

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance





def print_tree(estimator, features, class_names=None, filled=True):
    '''
    Creates images of tree models using pydot.
    '''
    tree = estimator
    names = features
    color = filled
    classn = class_names
    
    dot_data = StringIO()
    export_graphviz(estimator, out_file=dot_data, feature_names=features, class_names=classn, filled=filled)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    return graph


def calc_train_test_mse_decision_tree_regr(X_train, X_test, y_train, y_test, max_depth, cv):
    '''
    Fits and predicts a tree-based regression model for a range of depths.
    Uses cross validation to calculate the mean squared error for the training set.
    Calculates the mean squared error for the test set for each depth.
    
    Returns:
        train_c_val_df (df)      : All MSEs computer for all folds for each depth.
        avg_train_mse_arr (arr)  : The average MSE for the result of the cross validation.
        test_mse_arr (arr)       : The test MSEs for each depth.
    '''
    avg_train_mse_arr = []
    train_cross_val_df = pd.DataFrame()
    test_mse_arr = []
    
    for depth in range(1, max_depth+1):
        regr = DecisionTreeRegressor(max_depth=depth)
        regr.fit(X_train, y_train)
        
        # training set metrics
        train_cross_val_arr = -1 * cross_val_score(regr, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
        train_cross_val_df[str(depth) + '_depth'] = train_cross_val_arr

        train_mse = np.mean(train_cross_val_arr)
        avg_train_mse_arr.append(train_mse)

        # test set
        pred = regr.predict(X_test)
        test_mse = mean_squared_error(pred, y_test)
        test_mse_arr.append(test_mse)

    return train_cross_val_df, avg_train_mse_arr, test_mse_arr


def calc_train_test_mse_bagging_regr(X_train, X_test, y_train, y_test, max_depth, cv):
    '''
    Fits and predicts a tree-based regression model for a range of depths.
    Uses cross validation to calculate the mean squared error for the training set.
    Calculates the mean squared error for the test set for each depth.
    
    Returns:
        train_c_val_df (df)      : All MSEs computer for all folds for each depth.
        avg_train_mse_arr (arr)  : The average MSE for the result of the cross validation.
        test_mse_arr (arr)       : The test MSEs for each depth.
    '''
    avg_train_mse_arr = []
    train_cross_val_df = pd.DataFrame()
    test_mse_arr = []
    
    for depth in range(1, max_depth+1):
        regr = RandomForestRegressor(max_depth=depth, max_features=X_train.shape[1])
        regr.fit(X_train, y_train)
        
        # training set metrics
        train_cross_val_arr = -1 * cross_val_score(regr, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
        train_cross_val_df[str(depth) + '_depth'] = train_cross_val_arr

        train_mse = np.mean(train_cross_val_arr)
        avg_train_mse_arr.append(train_mse)

        # test set
        pred = regr.predict(X_test)
        test_mse = mean_squared_error(pred, y_test)
        test_mse_arr.append(test_mse)

    return train_cross_val_df, avg_train_mse_arr, test_mse_arr


def display_bagging_rf_feature_importance(estimator, X, plot_title, fig_size):
    '''
    Produces a horizontal bar plot summarizing the information gain by each feature.
    
    Estimator must be a fitted model.
    X must be a dataframe with the columns named as the features.
    
    '''
    feature_importance = pd.DataFrame()
    feature_importance['feature']=X.columns
    feature_importance['importance'] = estimator.feature_importances_*100
    feature_importance.sort_values('importance', axis=0, ascending=True, inplace=True)

    fig, ax = plt.subplots(figsize=fig_size)

    features = feature_importance['feature'].values
    scores = feature_importance['importance'].values

    ax.barh(features, scores)
    ax.set_title(plot_title, fontsize=15)
    ax.set_xlabel('Importance')
    fig.tight_layout();
    
    
    