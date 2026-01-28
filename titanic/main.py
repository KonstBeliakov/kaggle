# https://www.kaggle.com/competitions/titanic

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    # extract some information from name
    df["NameLength"] = df["Name"].str.len()
    df["Miss"] = df["Name"].str.contains("Miss.", case=False)
    df["Mr"] = df["Name"].str.contains("Mr.", case=False)
    df["Mrs"] = df["Name"].str.contains("Mrs.", case=False)

    df["AgeMissing"] = df["Age"].isnull()
    df["Age"] = df["Age"].fillna(df["Age"].median())

    df["CabinMissing"] = df["Cabin"].isnull()
    df["Embarked"] = df["Embarked"].fillna("Unknown")

    df = df.drop(columns=["Name", "Ticket", "Cabin"])

    # one-hot encoding
    df = pd.get_dummies(
        df,
        columns=["Sex", "Pclass", "Embarked", "Parch"],
        drop_first=True
    )
    return df

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=39
)

df_train = pd.read_csv('train.csv')
df_train = preprocess_df(df_train)

df_train.to_csv('train_processed.csv', index=False)

X = df_train.drop('Survived', axis=1)
train_columns = X.columns
y = df_train['Survived']

rf.fit(X, y)

df_test = pd.read_csv('test.csv')

df_test = preprocess_df(df_test)
df_test = df_test.reindex(columns=train_columns, fill_value=0)

predictions = rf.predict(df_test)

test_passenger_ids = df_test["PassengerId"].copy()
submission = pd.DataFrame(
    {
        "PassengerId": test_passenger_ids,
        "Survived": predictions
    }
)

submission.to_csv('submission.csv', index=False)
