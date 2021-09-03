import pandas as pd

def null_count_by_column(df):
    print(f'DataFrame shape: {df.shape}', end='\n\n')
    col_missingValues = (df.isnull().sum())
    print(col_missingValues[col_missingValues > 0])


def pivot_cat(df, columns):
    for col in columns:
        pivot = print(pd.crosstab(df['Survived'], df[col], values='Ticket', aggfunc='count', normalize='columns'))
        print(f'{pivot}')




