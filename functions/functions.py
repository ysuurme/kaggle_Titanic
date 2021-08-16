import pandas as pd

def null_count_by_column(df):
    print(f'DataFrame shape: {df.shape}', end='\n\n')
    col_missingValues = (df.isnull().sum())
    print(col_missingValues[col_missingValues > 0])


def pivot_cat(df, columns):
    for col in columns:
        pivot = print(pd.crosstab(df['Survived'], df_train[col], values='Ticket', aggfunc='count', normalize='columns'))
        print(f'{pivot}')

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
