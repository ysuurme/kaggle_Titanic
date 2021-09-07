import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


def enc_1hot(df, col):
    """Returns the dataframe with 1 hot encoding of provided columns"""
    df = pd.concat([df, pd.get_dummies(df[col], prefix= col)], axis=1)
    return df


def enc_map(df):
    """"Returns the dataframe with map encoding of provided columns"""
    col_cat = df.select_dtypes(['object']).columns
    encoding = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
                'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22,
                'X': 23, 'Y': 24, 'Z': 25}
    for col in col_cat:
        df[f'ord_{col}'] = df[col].map(encoding)
    return df


def enc_ord(df, cols):
    """"Returns the dataframe with ordinal encoding of provided columns"""
    enc = OrdinalEncoder(dtype='uint8')
    enc.fit(df[cols])
    df[f'ord_{col}'] = enc.transform(df[col])
    return df


def enc_label(df, cols):
    """"Returns the dataframe with label encoding of provided columns"""
    enc = LabelEncoder()
    for col in cols:
        df[f'lab_{col}'] = enc.fit_transform(df[col])
    return df


def enc_freq(df, cols):
    """"Returns the dataframe with frequency encoding of provided columns"""
    for col in cols:
        df[f'freq_{col}'] = df.groupby(col)[col].transform('count')
    return df


def imp_age(df):
    """Missing values in Age are filled with median age per Sex/Pclass group due to high correlation"""
    df_pivot = df.groupby(['Sex', 'Pclass']).median()['Age']
    df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
    for index, value in df_pivot.iteritems():
        print(f'Imputed Sex/Pclass: {index} - Age: {value}')
    return df


def imp_fare(df):
    """Missing values in Fare are filled with mean fare per Pclass group due to high correlation"""
    df_pivot = df.groupby(['Pclass']).mean()['Fare']
    df['Fare'] = df.groupby(['Pclass'])['Fare'].apply(lambda x: x.fillna(x.mean()))
    for index, value in df_pivot.iteritems():
        print(f'Imputed Pclass: {index} - Fare: {value}')
    return df
