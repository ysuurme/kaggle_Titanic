import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, \
    explained_variance_score

from functions.functions import *
from functions.feature_eng import *
from functions.plots import *

PLOT = False  # Mark plot is 'True' for updating the plots in folder figures
SEED = 1  # Random seed for model training
TRAIN = True  # Mark train is 'True' for training a model based on training data, mark 'False' for loading a pickle file
SAVE = True  # Mark save is 'True' for pickling the trained model with a timestamp, mark 'False for not saving the model

# Loading data as pandas dataframe:
df_train = pd.read_csv('sourceData/train.csv')  # Survival provided
df_test = pd.read_csv('sourceData/test.csv')  # Survival not provided, to predict
df_titanic = pd.read_csv('sourceData/titanic.csv')  # Full dataset for checking prediction accuracy
df_output = df_test[['PassengerId', 'Name']]

data = [df_train, df_test]

"""
A. Exploratory Data Analysis:
"""

# A.1 Data overview:
df_train.describe()
df_train.info()

df_train.drop(labels='PassengerId', axis=1, inplace=True)
df_test.drop(labels='PassengerId', axis=1, inplace=True)
# Conclusions:
# - 'PassengerId' is a mere row identifier which is dropped from the analysis

# A.2 Missing values
null_count_by_column(df_train)  # Print features for which values are null
null_count_by_column(df_test)

# Conclusions:
# - For 'Cabin' the majority of data points are missing, hence imputing data would probably not add value
# - For 'Age' and 'Fare' missing datapoints may be imputed
# - For 'Embarked' only 2 values are missing which will be dropped

df_train.dropna(subset=['Embarked'], inplace=True)  # Embarked contains only 2 rows with missing values
df_test.dropna(subset=['Embarked'], inplace=True)

df_train = imp_age(df_train)  # Impute median age based on Sex/Pclass groups
df_test = imp_age(df_test)

df_train = imp_fare(df_train)  # Impute mean fare based on Pclass groups
df_test = imp_fare(df_test)

# A.3 Target Distribution
surv = sum(df_train['Survived'])
surv_women = df_train.loc[df_train.Sex == 'female']["Survived"]
surv_men = df_train.loc[df_train.Sex == 'male']["Survived"]

print(f'{surv} from the {len(df_train)} training observations survived the Titanic, indicating a survival rate of '
      f'{surv / len(df_train):.2%}')

# Conclusions:
# - The minority of persons in the training set survived the Titanic

# A.4 Feature Target Distribution
print(df_train.dtypes)

# A.4.1 Continuous Features
col_cont = df_train.select_dtypes(include='float64')

for col in col_cont:
    q = pd.qcut(df_train[col], 10)
    print(df_train.pivot_table('Survived', index=q))

# Conclusions:
# - The Distribution of Age indicates that children (<16) have a higher survival rate
# - The Distribution of Fare indicates that a higher Fare (10.5+) indicates a higher survival rate

# A.4.2 Categorical Features
col_cat = df_train.select_dtypes(exclude='float64')

print(f'{surv} from the {len(df_train)} training observations survived the Titanic, from which {sum(surv_women)}'
      f' are female ({sum(surv_women) / surv:.2%}), and {sum(surv_men)} are male ({sum(surv_men) / surv:.2%})')
print(
    f'From the total number of woman {sum(surv_women) / len(surv_women):.2%} survived, and from the total number of men'
    f' {sum(surv_men) / len(surv_men):.2%} survived the Titanic')

pivot_cat(df_train, col_cat)  # print the survival rate per categorical feature category

# Conclusions:
# - Female passengers have a higher survival rate than male
# - Passengers embarked from Cherbourg (C) have the highest survival rate
# - Passengers travelling with Parents or Children have a higher survival rate
# - Passengers travelling class 1 or 2 have a higher survival rate than passengers travelling class 3
# - Passengers travelling with a Sibling or Spouse have a higher survival rate

# A.5 Correlation
print(f'Correlation Matrix: \n {df_train.corr().round(2)}')

# Conclusions:
# - Feature correlation indicates relationships which may be used for creating new features
# - Spikes in a distribution (f.i. 'Age') may be captured via a decision tree model
# - Ordinal relations (f.i. 'Pclass) may be captured via a linear model

# A.6 EDA Plots
if PLOT:
    plot_hist(df_train, col_cont)  # Plot A.4.1 Continuous Features
    plot_count(df_train, col_cat)  # Plot A.4.2 Categorical Features
    plot_corr(df_train)  # Plot A.5 Correlation

"""
B. Feature Engineering:
"""

# B.1. New features

# B.1.1 Creating new features from arithmetics
df_train['Family_Size'] = 1 + df_train['SibSp'] + df_train['Parch']
df_test['Family_Size'] = 1 + df_test['SibSp'] + df_test['Parch']

df_train['Cabin_n'] = df_train.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))  # Count Cabins, 0 is NaN
df_test['Cabin_n'] = df_test.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))

df_train['Cabin_section'] = df_train.Cabin.apply(lambda x: str(x)[0])  # Retrieve section, first Cabin character
df_test['Cabin_section'] = df_test.Cabin.apply(lambda x: str(x)[0])

df_train['Name_title'] = df_train.Name.apply(
    lambda x: x.split(',')[1].split('.')[0].strip())  # Retrieve title from Name
df_test['Name_title'] = df_test.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

df_train['freq_Ticket'] = df_train.groupby('Ticket')['Ticket'].transform('count')  # Indicates n people travelling
# with the same ticket
df_test['freq_Ticket'] = df_test.groupby('Ticket')['Ticket'].transform('count')

# B.1.2 Creating new features from Binning Continuous features
df_train['bin_Fare'] = pd.qcut(df_train['Fare'], 15)
df_test['bin_Fare'] = pd.qcut(df_test['Fare'], 15)

df_train['bin_Age'] = pd.qcut(df_train['Age'], 10)
df_test['bin_Age'] = pd.qcut(df_test['Age'], 10)

# B.3 Binary Encoding
df_train['bin_Sex'] = df_train.Sex.map({'male': 0, 'female': 1})  # Map binary 'Sex'
df_test['bin_Sex'] = df_test.Sex.map({'male': 0, 'female': 1})

# B.4 Frequency Encoding
encoding = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Large', 6: 'Large', 7: 'Large', 8: 'Large',
            9: 'Large', 10: 'Large', 11: 'Large', 12: 'Large'}  # Based on Countplot families of 2-4 persons are 'Small'
df_train[f'ord_Family_Size'] = df_train['Family_Size'].map(encoding)
df_test[f'ord_Family_Size'] = df_test['Family_Size'].map(encoding)

# B.5 Label Encoding
col_label = ['Embarked', 'bin_Fare', 'bin_Age', 'Cabin_section', 'Name_title', 'ord_Family_Size']
df_train = enc_label(df_train, col_label)
df_test = enc_label(df_test, col_label)

# B.6 1-Hot Encoding
df_train = enc_1hot(df_train, 'Embarked')  # one-hot encode categorical feature 'Embarked'
df_test = enc_1hot(df_test, 'Embarked')

# B.X Feature Engineering plots
if PLOT:
    plot_count(df_train, ['Family_Size'])  # Plot B.1.1 Creating new features from arithmetics
    plot_count(df_train, ['Cabin_n'])  # Plot B.1.1 Creating new features from arithmetics
    plot_count(df_train, ['Cabin_section'])  # Plot B.1.1 Creating new features from arithmetics
    plot_count(df_train, ['Name_title'])  # Plot B.1.1 Creating new features from arithmetics
    plot_count(df_train, ['bin_Fare'])  # Plot B.1.2 Creating new features from Binning Continuous features
    plot_count(df_train, ['bin_Age'])  # Plot B.1.2 Creating new features from Binning Continuous features

# Plotting the data
# plot_corr(df_train, col_num)  # Plot Correlation Heatmap for numerical features
# plot_mi_scores(mi_scores.head(20))  # Plot MI scores for numerical features

"""
C. Model:
"""

# C.1 Model Data selection
y = df_train['Survived']

feature_scope = df_test.select_dtypes(include=['float64', 'int64', 'int32', 'uint8']).columns
X = df_train[feature_scope]
X_test = df_test[feature_scope]  # Source data for predicting Titanic survivors

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=SEED)  # Setup train/test data

""""
The steps to building and using a model are:

Specify: Define the type of model that will be used, and the parameters of the model.
Fit: Capture patterns from provided data. This is the heart of modeling.
Predict: Predict the values for the prediction target (y)
Evaluate: Determine how accurate the model's predictions are.
"""

# C.2 Model specification
dir_model = 'modelsTrained/'

if TRAIN:

    # C.2.1 Random Forest Classifier
    model_Forest = RandomForestClassifier(max_depth=5, min_samples_split=4, min_samples_leaf=5, max_features='auto',
                                          random_state=SEED)

    # C.2.2 Model Selection
    model = model_Forest

    # C.2.3 Model Training
    model.fit(X_train, y_train)

else:
    filename_model = 'trained_modelForest.sav'
    filepath_model = os.path.join(dir_model, filename)
    model = pickle.load(open(filepath_model, 'rb'))
    print(f'Loading pickled model: {filepath_model}')

predictions_train = model.predict(X_valid)

# C.3 Model Evaluation (metrics)
rmse = mean_squared_error(y_true=y_valid, y_pred=predictions_train, squared=False)
print(f'Competition score metric! Root Mean Squared Error: {rmse:.4f}')  # Best possible score is: 0.0000
mae = mean_absolute_error(y_true=y_valid, y_pred=predictions_train)
print(f'Mean Absolute Error: {mae:.4f}')  # Best possible score is: 0.0000
mse = mean_squared_error(y_true=y_valid, y_pred=predictions_train, squared=True)
print(f'Mean Squared Error: {mse:.4f}')  # Best possible score is: 0.0000
mape = mean_absolute_percentage_error(y_true=y_valid, y_pred=predictions_train)
print(f'Mean Absolute Percentage Error: {mape:.2%}')  # Best possible score is: 0.00%
evs = explained_variance_score(y_true=y_valid, y_pred=predictions_train)
print(f'Explained Variance Score: {evs:.2f}')  # Best possible score is: 1.00

# Save model to disk (pickle model)
timestamp = f'{datetime.datetime.now():%d%m%y_%H%M}'

if SAVE:
    filename_model = f'trained_modelForest_{timestamp}.sav'
    filepath_model = os.path.join(dir_model, filename_model)
    pickle.dump(model, open(filepath_model, 'wb'))
    print(f'Saved pickled model: {filepath_model}')

# D. Output
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': df_output.PassengerId, 'Survived': predictions})

dir_output = 'outputData/'
filename_output = f'survivor_estimation_{timestamp}.csv'
filepath_output = os.path.join(dir_output, filename_output)
output.to_csv(filepath_output, index=False)
print(f'The survivor estimation of the test data was saved to {filepath_output}')

"""
# Compare the survivor estimation against actual survivor observations
df = pd.merge(df_test[['Name', 'PassengerId']], df_titanic[['name', 'survived']], left_on='Name', right_on='name',
              how='left') \
    .drop_duplicates(subset='PassengerId')
df2 = pd.merge(output.set_index('PassengerId'), df.set_index('PassengerId'), left_index=True, right_index=True,
               how='left')

df2['correct'] = df2['Survived'] == df2['survived']

# Calculate success rate of random forest survival estimation
success_rate = sum(df2['correct']) / len(df2['correct'])
print(f'The random forest predicted {success_rate:.2%} survivors correctly!')
"""