
import pandas as pd

df = pd.read_csv("C:\\Users\\ASUS\\Downloads\\archive (1)\\Titanic-Dataset.csv")

print(df)

print(df.head(3))

print(df.tail())

print(df.sample(5, random_state=42))

print(df.shape)

print(df.columns)

print(df.dtypes)

df.info()

print(df.describe())

print(df.describe(include='all'))

print(df['Sex'].value_counts())

print(df['Sex'].unique())

print(df.isna().sum())

print(df.duplicated().sum())

df['Age'] = df['Age'].fillna(df['Age'].median())

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df['Cabin'] = df['Cabin'].fillna('Unknown')

df = df.drop_duplicates()

print(df.isna().sum())

print(df)

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

df['Alone'] = df['FamilySize'].apply(lambda x: 1 if x == 1 else 0)

df['Survival_Status'] = df['Survived'].map({0: 'Not Survived', 1: 'Survived'})

print(df.head())

print(df['Survival_Status'].value_counts())

print(df.groupby('Sex')['Survived'].mean())

print(df.groupby('Pclass')['Survived'].mean())

print(df.groupby('Embarked')['Survived'].mean())

print(df.groupby('Sex')['Age'].mean())

print(df.sort_values(by='Age'))

print(df.sort_values(by='Fare', ascending=False))

print(df.corr(numeric_only=True))

df.to_csv("Cleaned_Titanic_Dataset.csv", index=False)

print("Data cleaning and analysis completed successfully")
