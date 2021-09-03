import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split

from functions.functions import *
from functions.feature_eng import *

# Loading data as pandas dataframe:
df_train = pd.read_csv('sourceData/train.csv')  # Survival provided
df_test = pd.read_csv('sourceData/test.csv')  # Survival not provided, to predict
df_titanic = pd.read_csv('sourceData/titanic.csv')  # Full dataset for checking prediction accuracy

data = [df_train, df_test]

# Understanding the Data:
df_train.describe()
df_train.info()

"""
Data preparation:
"""

df_train = enc_1hot(df_train, 'Embarked')  # one-hot encode categorical feature 'Embarked'
df_test = enc_1hot(df_test, 'Embarked')

df_train.dropna(subset=['Embarked'], inplace=True)  # Embarked contains only 2 rows with missing values
df_test.dropna(subset=['Embarked'], inplace=True)

df_train['bin_Sex'] = df_train.Sex.map({'male': 0, 'female': 1})  # Map binary 'Sex'
df_test['bin_Sex'] = df_test.Sex.map({'male': 0, 'female': 1})

df_train['Cabin_n'] = df_train.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))  # Count Cabins, 0 is NaN
df_test['Cabin_n'] = df_test.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))

df_train['Cabin_section'] = df_train.Cabin.apply(lambda x: str(x)[0])  # Retrieve section, first Cabin character
df_test['Cabin_section'] = df_test.Cabin.apply(lambda x: str(x)[0])

df_train['Name_title'] = df_train.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())  # Retrieve title from Name
df_test['Name_title'] = df_test.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

enc = OrdinalEncoder(dtype='uint8', handle_unknown='use_encoded_value', unknown_value=99)
enc.fit(df_train[['Name_title', 'Cabin_section']])
df_train[['ord_Name_title', 'ord_Cabin_section']] = enc.transform(df_train[['Name_title', 'Cabin_section']])  # Ordinally encode person Title and Cabin Section
df_test[['ord_Name_title', 'ord_Cabin_section']] = enc.transform(df_test[['Name_title', 'Cabin_section']])

df_train = imp_age(df_train)  # Impute median age based on Sex/Pclass groups
df_test = imp_age(df_test)

df_train = imp_fare(df_train)  # Impute mean fare based on Pclass groups
df_test = imp_fare(df_test)


"""
Exploratory Data Analysis:
"""
col_cat = list(df_train.select_dtypes(['object']).columns)  # Categorical columns are of type object
col_num = list(df_train.select_dtypes(['float64', 'int64']).columns)  # Numerical columns are of type float/integer 64 bit
col_ord = list(df_train.select_dtypes(['uint8']).columns)  # Ordinally encoded columns are of type integer 8 bit

null_count_by_column(df_train)  # Print features for which values are null
null_count_by_column(df_test)

pd.pivot_table(data=df_train[col_num], index='Survived')  # todo plot relation with Survived
pivot_cat(df_train, col_cat)

# Plotting the data
# plot_hist(df_train, col_ord)  # Plot Histograms for ordinal features
# plot_corr(df_train, col_num)  # Plot Correlation Heatmap for numerical features
# plot_mi_scores(mi_scores.head(20))  # Plot MI scores for numerical features
# plot_count(df_train, col_cat)  # Plot Count for categorical features


"""
Feature Engineering:
"""

# Separate target from features
y = df_train['Survived']
X = df_train.drop(['Survived'], axis=1)
X_test = df_test

X_train, X_test = df_train, df_test









"""
Feature engineering:
"""





print(f'Conclusion 1: "Cabin" is a NON relevant feature as the majority of observations are missing \n')



# Plot Histogram for numerical features
for col in col_num:
      plt.clf()
      sns.histplot(x=col, data=df_train, hue='Survived', multiple='stack', palette=['red', 'blue'])
      plt.title(f'Survival Histogram: {col}')
      plt.xlabel(col)
      plt.ylabel('Survived No (0) Yes (1)')
      plt.show()
      plt.savefig(f'figures/EDA_Hist_{col}.png')

# Plot numerical features correlation
print(df_train[col_num].corr())
sns.heatmap(df_train[col_num].corr())
print(pd.pivot_table(df_train, index='Survived', values=col_num))

# Plot countplot for categorical features
for col in col_cat:
      plt.clf()
      sns.countplot(x=col, data=df_train, hue='Survived', palette=['red', 'blue'])
      plt.title(f'Category count: {col}')
      plt.xlabel(col)
      plt.ylabel('Survived No (0) Yes (1)')
      plt.show()
      plt.savefig(f'figures/EDA_Count_{col}.png')


# Determine gender based survival rate based from the 891 training observations
surv = sum(df_train['Survived'])
surv_women = df_train.loc[df_train.Sex == 'female']["Survived"]
surv_men = df_train.loc[df_train.Sex == 'male']["Survived"]

print(f'{surv} from the {len(df_train)} training observations survived the Titanic, from which {sum(surv_women)}'
      f' are female ({sum(surv_women)/surv:.2%}), and {sum(surv_men)} are male ({sum(surv_men)/surv:.2%})')

rate_women = sum(surv_women)/len(surv_women)
rate_men = sum(surv_men)/len(surv_men)

print(f'From the total number of woman {rate_women:.2%} survived, and from the total number of men {rate_men:.2%}'
      f' survived the Titanic')

# Plot survivors based on sex:
_ = sns.countplot(x='Survived', data=df_train, hue='Sex')
_ = plt.title('Titanic survivors by Sex')
_ = plt.xlabel('Survived No (0) Yes (1)')
_ = plt.ylabel('n Survivors')
plt.show()
print(f'Conclusion 2: "Sex" is A relevant feature as women seem more likely to survive the Titanic than men \n')

# Plot survivors based on age:
_ = sns.histplot(x='Age', data=df_train, hue='Survived', multiple='stack', palette=['red', 'blue'])
_ = plt.title('Titanic survivors by Age')
_ = plt.xlabel('Age')
_ = plt.ylabel('Survived No (0) Yes (1)')
plt.show()


print(f'Conclusion 3: "Age" is A relevant feature as younger people seem more likely to survive the Titanic\n')

# Price Class relation to Survival
print(df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',
                                                                                              ascending=False))
print(f'Conclusion 4: "Pclass" is A relevant feature as traveling a higher class seems to indicate that it is more'
      f' likely to survive the Titanic\n')

# Sibling relation to Survival
print(df_train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived',
                                                                                            ascending=False))
print(f'Conclusion 5: "SibSp" is A relevant feature as having less siblings/spouses seems to indicate that it is more'
      f' likely to survive the Titanic\n')

# Parent-child relation to survival
print(df_train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived',
                                                                                            ascending=False))
print(f'Conclusion 6: "Parch" is A relevant feature as traveling with parents-child seems to indicate that it is more'
      f' likely to survive the Titanic\n')


""""
The steps to building and using a model are:

Specify: Define the type of model that will be used, and the parameters of the model.
Fit: Capture patterns from provided data. This is the heart of modeling.
Predict: Predict the values for the prediction target (y)
Evaluate: Determine how accurate the model's predictions are.
"""

model = RandomForestClassifier(n_estimators=100, random_state=1)
# Train the random forest based on df features
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_C", "Embarked_Q", "Embarked_S"]

X = X_train[features]
y = X_train["Survived"]

model.fit(X, y)
predictions = model.predict(X_test[features])

fileName = 'outputData/survivor_estimation.csv'
output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predictions})
output.to_csv(fileName, index=False)
print(f'The survivor estimation of the test data was saved to {fileName}')

# Compare the survivor estimation against actual survivor observations
df = pd.merge(df_test[['Name', 'PassengerId']], df_titanic[['name', 'survived']], left_on='Name', right_on='name', how='left')\
      .drop_duplicates(subset='PassengerId')
df2 = pd.merge(output.set_index('PassengerId'), df.set_index('PassengerId'), left_index=True, right_index=True, how='left')

df2['correct'] = df2['Survived']==df2['survived']

# Calculate success rate of random forest survival estimation
success_rate = sum(df2['correct'])/len(df2['correct'])
print(f'The random forest predicted {success_rate:.2%} survivors correctly!')