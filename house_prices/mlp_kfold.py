# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_FRACTION),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_FRACTION),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.network(x)

    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

        for epoch in range(EPOCHS):
            self.train()
            optimizer.zero_grad()
            outputs = self.forward(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f'\t{epoch + 1} epoch completed. Loss: {loss.item():.4f}')


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
        #'Alley',  # low diversity of values
        'Utilities',  # in training data all values are the same
        #'LandContour',  # low diversity of values
        #'LandSlope',  # low diversity of values
        #"RoofMatl",  # low diversity of values
        #"ExterCond",  # low diversity of values
        #"BsmtCond",
        #"BsmtExposure",
        "Heating",  # all values are the same
        #"GarageQual",
        #"GarageCond",
        #"PavedDrive",
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

DROPOUT_FRACTION = 0.05
EPOCHS = 500
LEARNING_RATE = 0.0002
K = 5

if __name__ == '__main__':
    train_df = pd.read_csv('train.csv')

    train_df = preprocess(train_df)

    X = train_df.drop(['SalePrice', 'Id'], axis=1)
    X_columns = X.columns
    X = torch.tensor(X.values, dtype=torch.float32)
    Y = train_df['SalePrice']
    Y = torch.tensor(Y.values, dtype=torch.float32)

    kf = KFold(n_splits=K, shuffle=True, random_state=39)
    fold_scores = []

    models = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        model = MLP(input_size=X.shape[1])
        models.append(model)

        print(f'Training {fold + 1} model...')
        model.fit(X_train, Y_train)
        print('done')

        model.eval()
        with torch.no_grad():
            val_tensor = torch.tensor(X_val, dtype=torch.float32)
            predictions = model(val_tensor).squeeze()
            mse = torch.nn.functional.mse_loss(predictions, torch.tensor(Y_val, dtype=torch.float32))
            rmse = torch.sqrt(mse).item()
            fold_scores.append(rmse)

    print(f'Average RMSLE: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})')

    test_df = pd.read_csv('test.csv')
    test_df_ids = test_df['Id']
    test_df = test_df.drop('Id', axis=1)
    test_df = preprocess(test_df)
    test_df = test_df.reindex(columns=X_columns, fill_value=0)

    predictions = []
    for model in models:
        model.eval()

        X_test_tensor = torch.tensor(test_df.values, dtype=torch.float32)

        with torch.no_grad():
            predictions.append(model.forward(X_test_tensor).squeeze())

    avg = []
    for id in range(X_test_tensor.shape[0]):
        s = 0
        for model_idx in range(K):
            s += predictions[model_idx][id]
        avg.append(float((s / K).item()))

    submission = pd.DataFrame(
        {
            'Id': test_df_ids.astype('int'),
            'SalePrice': pd.DataFrame(avg).squeeze()
        }
    )
    submission.to_csv('submission.csv', index=False)
