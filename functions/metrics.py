import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_regression

FOLDS = 5  # Cross validation folds used


def model_cv_score(model, x, y, cv=FOLDS):
    """Model mean Cross Validation score"""
    model_name = type(model).__name__
    cv_score = cross_val_score(model, x, y, cv=cv)
    print(
        f'{model_name} - #{cv} fold Cross Validation: Mean - {cv_score.mean():.2%} | Min - {np.min(cv_score):.2%} | Max - {np.max(cv_score):.2%}')


def mi_score(X, y):
    """"Mutual information (MI) between two random variables is a non-negative value, which measures the dependency
        between the variables. It is equal to zero if and only if two random variables are independent, and higher
        values mean higher dependency."""
    X = X.select_dtypes(exclude=['object'])  # All features should now have numerical dtypes
    mi_scores = mutual_info_regression(X, y, random_state=1)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    mi_scores = mi_scores.rename('mi_scores')
    return mi_scores


def var_score(X):
    """"Higher variability per feature may indicate more explanatory value"""
    X = X.select_dtypes(exclude=['object'])  # All features should now have numerical dtypes
    var_scores = X.var()
    var_scores = var_scores.sort_values(ascending=False)
    var_scores = var_scores.rename('var_scores')
    return var_scores


def model_score(models, x, y):
    for model in models:
        model_name = type(model).__name__
        model.fit(x, y)
        score = model.score(x, y)
        print(f'{model_name} Score: {score:.3f}')
        cv = cross_val_score(model, x, y)
        print(f'{model_name} Cross Validation: {cv.mean():.3f}')
    return models
