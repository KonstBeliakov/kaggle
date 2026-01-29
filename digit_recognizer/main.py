# https://www.kaggle.com/competitions/digit-recognizer/overview
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


HIDDEN_LAYER_SIZE = 256
EPOCHS = 30
K = 5 # number of models in KFold


# later I need to change that to CNN
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_LAYER_SIZE, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

    def fit(self, x: torch.Tensor, y: pd.DataFrame):
        # we dont need to split x and y into train and val, because we are using KFolding in main

        print(f'{type(y)} {y=}')
        labels = y['label']
        indices = y.drop('label', axis=1)

        # we need to transform y into list of 10 probabilities (1 equal to 1 and others equal to 0)
        # For each label we have tensor of size 10
        target_probabilities = torch.Tensor([[int(target==i) for i in range(10)] for target in labels])

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(EPOCHS):
            self.train()
            optimizer.zero_grad() # forgetting the previous gradient
            outputs = self.forward(x)
            loss = criterion(outputs, target_probabilities)
            loss.backward()
            optimizer.step() # updating weights of out network

            if (epoch + 1) % 20 == 0:
                print(f'\tEpoch {epoch + 1} finished')


if __name__ == '__main__':
    train_df = pd.read_csv('train.csv')

    X = train_df.drop('label', axis=1)
    Y = train_df['label']

    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    Y_tensor = torch.tensor(Y.values, dtype=torch.float32)

    models: list[MLP] = []

    kf = KFold(n_splits=K, shuffle=True, random_state=39)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X.columns)):
        #print(f'{fold=} {train_idx=} {val_idx=}')
        X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
        y_train, y_val = Y[train_idx], Y[val_idx]
        print(type(y_train), type(y_val), f'{y_train=}\n {y_val=}')

        model = MLP(input_size=X.shape[1])
        models.append(model)

        print(f'Training {fold + 1}...')
        model.fit(X_train, pd.DataFrame(y_train))
        print('done')

        model.eval()
        with torch.no_grad():
            val_tensor = torch.tensor(X_val, dtype=torch.float32)
            predictions = model.forward(val_tensor).squeeze()
            y_val_tensor = torch.Tensor([[int(i==target) for i in range(10)] for target in y_val])
            mse = torch.nn.functional.mse_loss(predictions, y_val_tensor)
            rmse = torch.sqrt(mse).item()
            fold_scores.append(rmse)
            print(f"Fold {fold + 1} RMSLE: {rmse: 4f}")

        print(f'\n\nAverage RMSLE: {np.mean(fold_scores):.4f} (+/-{np.std(fold_scores):.4f})')

    test_df = pd.read_csv('test.csv')

    X_test = torch.tensor(test_df.values, dtype=torch.float32)

    predictions = []
    with torch.no_grad():
        for model in models:
            predictions.append(model.forward(X_test))
            print(type(predictions), predictions)
    print(type(predictions), predictions)

    # we need to calculate average predictions of test values by all models

    #sum_prediction = [[0] * 10 for _ in range(X.shape[0])]

    # for i in range(K):
    #     for j in range(X.shape[0] - 1):
    #         for k in range(10):
    #             sum_prediction[j][k] += predictions[i][j][k].item()

    stacked_predictions = torch.stack(predictions) # creating a tensor from list of tensors
    avg_predictions = torch.mean(stacked_predictions, dim=0) # calculating average (redusing number of dimensions of tensor by 1)
    labels = torch.argmax(avg_predictions, dim=1)  # finding maximal number in every row of size 10

    # and find the most probable digit for each ImageId

    submission = pd.DataFrame(
        {
            'ImageId': 0,
            'Label': labels
        }
    )

    submission.to_csv('submission.csv', index=False)
