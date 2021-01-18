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
from sklearn.ensemble import BaggingClassifier, BaggingRegressor 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
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


def display_bagging_rf_feature_importance(estimator, X, plot_title, fig_size, max_features):
    '''
    Produces a horizontal bar plot summarizing the information gain by each feature.
    
    Estimator must be a fitted model.
    X must be a dataframe with the columns named as the features.
    
    '''
    feature_importance = pd.DataFrame()
    feature_importance['feature']=X.columns
    feature_importance['importance'] = estimator.feature_importances_*100
    feature_importance.sort_values('importance', axis=0, ascending=True, inplace=True)
    feature_importance = feature_importance.iloc[:max_features, :]
    
    fig, ax = plt.subplots(figsize=fig_size)

    features = feature_importance['feature'].values
    scores = feature_importance['importance'].values

    ax.barh(features, scores)
    ax.set_title(plot_title, fontsize=15)
    ax.set_xlabel('Importance')
    fig.tight_layout();
    
def create_sorted_permutation_importance_df(model, X, y, n_iterations, random_state):
    '''
    Inputs:
    Enter a fitted tree-based model, e.g. Random Forest or Gradient Boost.
    X and y can be the training set or a hold-out set, i.e., validation or test.
    X and y must be dataframes.
    
    Return:
    Returns a sorted dataframe comprised of each feature's importance for the number of selected iterations.
    The dataframe is sorted by the features' medians.
    
    Misc:
    Permutation importance is calculated by the difference between a baseline metric
    and the resulting "score" when the feature is excluded.
    The larger the value, the greater the importance.
    Occasionally see negative values for permutation importances. 
    In those cases, the predictions on the shuffled (or noisy) data happened 
    to be more accurate than the real data. 
    This happens when the feature didn't matter (should have had an importance close to 0),
    but random chance caused the predictions on shuffled data to be more accurate.
    '''
    
    results = permutation_importance(model, X, y, n_repeats=n_iterations,
                                     random_state=random_state, n_jobs=-1)
    
    df = pd.DataFrame(results.importances.T * 100, columns=X.columns)
    sorted_idx = df.median().sort_values().index[::-1]
    sorted_df = df[sorted_idx]
    return sorted_df

def plot_horizontal_permutation_importance_boxplot(df, figsize, max_features):
    '''
    df must be sorted.
    max_features is a limit on the number of features displayed
    '''
    
    df2 = df.iloc[:, :max_features].copy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.boxplot(data=df2, orient='h', ax=ax)
    ax.set_xlabel('Importance')
    ax.set_title('Permutation Importance', fontsize=16)
    fig.tight_layout()

def calculate_and_plot_permutation_importance(model, X, y, n_iterations, random_state, figsize, max_features):
    '''
    Inputs:
    Enter a fitted tree-based model, e.g. Random Forest or Gradient Boost.
    X and y can be the training set or a hold-out set, i.e., validation or test.
    X and y must be dataframes.
    max_features is the number of features to be displayed on the boxplot.
    
    Returns a horizontal box plot summarizing each feature's permutation importance.
    '''
    sorted_df = create_sorted_permutation_importance_df(model, X, y, n_iterations, random_state)
    plot_horizontal_permutation_importance_boxplot(sorted_df, figsize, max_features)
    