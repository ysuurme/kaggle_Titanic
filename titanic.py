import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from functions.functions import null_count_by_column, pivot_cat

# Loading data as pandas dataframe:
df_train = pd.read_csv('sourceData/train.csv')  # Survival provided
df_test = pd.read_csv('sourceData/test.csv')  # Survival not provided, to predict
df_titanic = pd.read_csv('sourceData/titanic.csv')  # Full dataset for checking prediction accuracy

# Understanding the Data:
# df_train.describe()
# df_train.info()

null_count_by_column(df_train)  # Print features for which values are null
null_count_by_column(df_test)  #todo consider = .Age/Fare.fillna(df.Age/Fare.mean()

null_count_by_column(X_test)  #todo consider = .Age/Fare.fillna(df.Age/Fare.mean()

col_num = ['Age', 'SibSp', 'Parch', 'Fare']
col_cat = ['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']
# col_cat = list(df_train.select_dtypes(['object']).columns)  #todo assign based on dtype

"""
Clean the Data:
"""
X_train, X_test = df_train, df_test
data = [X_train, X_test]
for df in data:

      df.dropna(subset=['Embarked'], inplace=True)  # Embarked contains only 2 rows with missing values

      df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # Sex to 0-1

      df['Cabin_n'] = df.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))  # 0 is NaN

      df['Cabin_section'] = df.Cabin.apply(lambda x: str(x)[0])

      df['Name_title'] = df.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

# Generate normally distributed values for "Age"
      age_mean, age_std = df["Age"].mean(), df["Age"].std()
      age_rand = np.random.normal(age_mean, age_std, df["Age"].isnull().sum()).astype(float)
      age_rand = np.ndarray.round(np.sqrt(age_rand**2))  # only positive age rounded up
      mask = df.loc[df['Age'].isnull()]
      df.loc[mask.index, 'Age'] = age_rand

# Generate mean value for missing "Fare"
      fare_mean = df['Fare'].mean()
      mask = df.loc[df['Fare'].isnull()]
      df.loc[mask.index, 'Fare'] = fare_mean

# OneHotEncoding Embarked
# Apply one-hot encoder to each column with categorical data
OH_encode_cols = ['Embarked']
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[OH_encode_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[OH_encode_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_test.index = X_test.index

#Label columns
OH_cols_train = OH_cols_train.rename(columns={0: 'Embarked_C', 1: 'Embarked_Q', 2: 'Embarked_S'})
OH_cols_test = OH_cols_test.rename(columns={0: 'Embarked_C', 1: 'Embarked_Q', 2: 'Embarked_S'})

# Add one-hot encoded columns to numerical features
X_train = pd.concat([X_train, OH_cols_train], axis=1)
X_test = pd.concat([X_test, OH_cols_test], axis=1)


# Determine one-hot or Ordinal encoding based on 'cardinality'
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))
# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])

ord_encoder = OrdinalEncoder()  #todo ordinal encoding
label_X_train[good_label_cols] = ord_encoder.fit_transform(X_train[good_label_cols])
label_X_valid[good_label_cols] = ord_encoder.transform(X_valid[good_label_cols])

"""
Explore the Data:
"""

pd.pivot_table(df_train, index='Survived', values=col_num)
pivot_cat(df_train, ['Pclass', 'Sex', 'Embarked', 'Cabin_n', 'Cabin_section', 'Name_title'])

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