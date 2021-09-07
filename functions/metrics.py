def mi_score(X, y):
    """"Mutual information (MI) between two random variables is a non-negative value, which measures the dependency
        between the variables. It is equal to zero if and only if two random variables are independent, and higher
        values mean higher dependency."""
    X = X.select_dtypes(exclude=['object'])  # All features should now have numerical dtypes
    mi_scores = mutual_info_regression(X, y, random_state=0)
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