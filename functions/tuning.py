def min_forest_mae(leaf_nodes, train_X, val_X, train_y, val_y):
    """Returns the node yielding the minimal Mean Absolute Error (MAE) for given training and validation data"""
    dict_mae = {}
    for node in leaf_nodes:
        model = DecisionTreeRegressor(max_leaf_nodes=node, random_state=1)
        model.fit(train_X, train_y)
        preds_val = model.predict(val_X)
        mae = mean_absolute_error(val_y, preds_val)
        dict_mae[node] = mae
    node = min(dict_mae, key=dict_mae.get)
    print(f'Best Fit Node modelTree: {node} - MAE: {dict_mae[node]:.0f}')
    return node

def min_tree_mae(leaf_nodes, train_X, val_X, train_y, val_y):
    """Returns the node yielding the minimal Mean Absolute Error (MAE) for given training and validation data"""
    dict_mae = {}
    for node in leaf_nodes:
        model = DecisionTreeRegressor(max_leaf_nodes=node, random_state=1)
        model.fit(train_X, train_y)
        preds_val = model.predict(val_X)
        mae = mean_absolute_error(val_y, preds_val)
        dict_mae[node] = mae
    node = min(dict_mae, key=dict_mae.get)
    print(f'Best Fit Node modelTree: {node} - MAE: {dict_mae[node]:.0f}')
    return node


def min_k_rmse(list_k, model, X_train, X_test, y_train, y_test):
    """Returns the # k best features yielding the minimal Root Mean Squared Error (RMSE) for given training and
    validation data"""
    dict_rmse = {}
    for k in list_k:
        SelectKBest(f_regression, k=k).fit_transform(X_train, y_train)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = mean_squared_error(y_test, predictions)
        dict_rmse[k] = rmse
    k_best = min(dict_rmse, key=dict_rmse.get)
    print(f'Best Fit k features modelLinear: {k_best} - RMSE: {dict_rmse[k_best]:.4f}')
    return dict_rmse


def var_threshold(df, threshold=0):
    """Returns features having threshold level in all samples"""
    varModel = VarianceThreshold(threshold=threshold)  # identifies features having >99% of same value in all samples
    varModel.fit(df)
    constVar = varModel.get_support()
    constCol = [col for col in df.columns if col not in df.columns[constVar]]
    return constCol