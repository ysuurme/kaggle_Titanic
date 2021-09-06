import pandas as pd


def null_count_by_column(df):
    print(f'DataFrame shape: {df.shape}', end='\n\n')
    col_missing_values = (df.isnull().sum())
    print(col_missing_values[col_missing_values > 0])


def pivot_cat(df, cols):
    """Print Pivot for categorical features in relation to 'Survived'."""
    for col in cols:
        if df[col].nunique() < 32:  # only plot categorical features with less than 32 unique values
            pivot = pd.crosstab(df['Survived'], df[col], values='Ticket', aggfunc='count', normalize='columns')
            print(f'{pivot}')
        else:
            print(f'No pivot created as #{df[col].nunique()} unique values retrieved for Categorical Feature: {col}')


