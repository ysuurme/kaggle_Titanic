import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Read train.csv as Pandas Dataframe
df_train = pd.read_csv('sourceData/train.csv')

# Read test.csv as Pandas Dataframe
df_test = pd.read_csv('sourceData/test.csv')

# Read titanic.csv as Pandas Dataframe
df_titanic = pd.read_csv('sourceData/titanic.csv')

# Print features for which values are NaN
for col in df_train.columns:
      n_missing = df_train[col].isnull().sum()
      if n_missing > 0:
            print(f'df_train[{col}] contains #{n_missing} missing values!')
print(f'Conclusion 1: "Cabin" is a NON relevant feature as the majority of observations are missing \n')

col_num = ['Age', 'SibSp', 'Parch', 'Fare']
col_cat = ['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']

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

# Generate normally distributed values for "Age"
age_mean = df_train["Age"].mean()
age_std = df_train["Age"].std()

# Compute random numbers between the mean, std and is_null
data = [df_train, df_test]
for d in data:
      age_rand = np.random.normal(age_mean, age_std, d["Age"].isnull().sum()).astype(float)
      age_rand = np.ndarray.round(np.sqrt(age_rand**2))  # only positive age rounded up

      mask = d.loc[d['Age'].isnull()]
      d.loc[mask.index, 'Age'] = age_rand
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

# Train the random forest based on df features
y = df_train["Survived"]

features = ["Sex", "Age", "SibSp", "Parch", "Pclass"]
x_train = pd.get_dummies(df_train[features])
x_test = pd.get_dummies(df_test[features])

model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=1)
model.fit(x_train, y)
predictions = model.predict(x_test)

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