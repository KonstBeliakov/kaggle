# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.hidden_size = 256
        self.network = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

    def forward(self, x):
        return self.network(x)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.0003)

        epochs = 1000
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self.forward(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                self.eval()
                with torch.no_grad():
                    val_outputs = self.forward(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')



def preprocess(df):
    numerical_features = ['MSSubClass', 'LotFrontage', 'LotArea', "OverallQual", "OverallCond", "YearBuilt",
                          "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
                          "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BedroomAbvGr",
                          "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr",
                          "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars", "GarageArea",
                          "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea",
                          "YrSold", ]
    categorical_features = ['MSZoning', 'Street', 'Alley', "LandContour", "LotConfig", "Neighborhood", "Condition1",
                            "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st",
                            "Exterior2nd", "LotShape",
                            "RoofStyle", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "Foundation",
                            "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "HeatingQC",
                            "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType",
                            "FireplaceQu", "GarageType", "GarageFinish", "PoolQC", "Fence", "SaleType", "SaleCondition"]
    drop_columns = [
        'Street',  # in the training data all values are the same
        'Alley',  # low diversity of values
        'Utilities',  # in training data all values are the same
        'LandContour',  # low diversity of values
        'LandSlope',  # low diversity of values
        "RoofMatl",  # low diversity of values
        "ExterCond",  # low diversity of values
        "BsmtCond",
        "BsmtExposure",
        "Heating",  # all values are the same
        "GarageQual",
        "GarageCond",
        "PavedDrive",
        "MiscFeature",  # hard to interpret
        "MiscVal",
    ]

    df = df.replace("NA", pd.NA)

    df = df.drop(columns=drop_columns)
    #df = pd.get_dummies(df, columns=list(set(categorical_features) - set(drop_columns)), drop_first=True)

    cat_cols = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    df = df.fillna(0.0)  # change that later

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.astype("float32")

    return df


if __name__ == '__main__':
    train_df = pd.read_csv('train.csv')

    train_df = preprocess(train_df)

    X = train_df.drop(['SalePrice', 'Id'], axis=1)
    Y = train_df['SalePrice']

    mlp = MLP(input_size=X.shape[1])
    mlp.fit(X, Y)

    test_df = pd.read_csv('test.csv')
    test_df_ids = test_df['Id']
    test_df = test_df.drop('Id', axis=1)
    test_df = preprocess(test_df)
    test_df = test_df.reindex(columns=X.columns, fill_value=0)

    mlp.eval()

    X_test_tensor = torch.tensor(test_df.values, dtype=torch.float32)

    with torch.no_grad():
        predictions = mlp.forward(X_test_tensor).squeeze()

    submission = pd.DataFrame(
        {
            'Id': test_df_ids.astype('int'),
            'SalePrice': predictions
        }
    )
    submission.to_csv('submission.csv', index=False)
