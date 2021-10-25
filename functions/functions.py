import pandas as pd


def null_count_by_column(df):
    """Lists number of missing values per column if n missing values > 0"""
    print(f'DataFrame shape: {df.shape}', end='\n\n')
    for col in df_train.columns:
        n_missing = df_train[col].isnull().sum()
        if n_missing > 0:
            print(f'df_train[{col}] contains #{n_missing} missing values!')


def pivot_cat(df, cols):
    """Print Pivot for categorical features in relation to 'Survived'."""
    for col in cols:
        if df[col].nunique() < 32:  # only plot categorical features with less than 32 unique values
            pivot = pd.crosstab(df['Survived'], df[col], values='Ticket', aggfunc='count', normalize='columns')
            print(f'{pivot}')
        else:
            print(f'No pivot created as #{df[col].nunique()} unique values retrieved for Categorical Feature: {col}')


