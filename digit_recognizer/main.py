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
EPOCHS = 100
DROPOUT_FRACTION = 0.2
LEARNING_RATE = 0.001
K = 5 # number of models in KFold


# later I need to change that to CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.fc = nn.Sequential(
            nn.LazyLinear(512), #nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv1(x.view(-1, 1, 28, 28))
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def fit(self, x: torch.Tensor, y: pd.DataFrame):
        # we dont need to split x and y into train and val, because we are using KFolding in main

        labels = y['label']

        # we need to transform y into list of 10 probabilities (1 equal to 1 and others equal to 0)
        # For each label we have tensor of size 10
        target_probabilities = torch.Tensor([[int(target==i) for i in range(10)] for target in labels])

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

        for epoch in range(EPOCHS):
            self.train()
            optimizer.zero_grad() # forgetting the previous gradient
            outputs = self.forward(x)
            loss = criterion(outputs, target_probabilities)
            loss.backward()
            optimizer.step() # updating weights of out network

            if (epoch + 1) % 10 == 0:
                print(f'\tEpoch {epoch + 1} finished')


if __name__ == '__main__':
    train_df = pd.read_csv('train.csv')

    X = train_df.drop('label', axis=1)
    Y = train_df['label']

    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    Y_tensor = torch.tensor(Y.values, dtype=torch.float32)

    models: list[CNN] = []

    kf = KFold(n_splits=K, shuffle=True, random_state=39)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X.columns)):
        X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
        y_train, y_val = Y[train_idx], Y[val_idx]

        model = CNN()
        models.append(model)

        print(f'Training {fold + 1}...')
        model.fit(X_train, pd.DataFrame(y_train))
        print('done')

        model.eval()
        with torch.no_grad():
            X_val = X_val.clone().detach().to(torch.float32)
            predictions = model.forward(X_val).squeeze()
            y_val_tensor = torch.Tensor([[int(i==target) for i in range(10)] for target in y_val])
            mse = torch.nn.functional.mse_loss(predictions.detach().clone(), y_val_tensor.detach().clone())
            rmse = torch.sqrt(mse).item()
            fold_scores.append(rmse)
            print(f"Fold {fold + 1} RMSLE: {rmse: 4f}")

        print(f'\n\nAverage RMSLE: {np.mean(fold_scores):.4f} (+/-{np.std(fold_scores):.4f})')

    test_df = pd.read_csv('test.csv')

    X_test = torch.tensor(test_df.values, dtype=torch.float32)
    test_items = X_test.shape[0]

    with torch.no_grad():
        predictions = [model.forward(X_test) for model in models]

    # we need to calculate average predictions of test values by all models
    stacked_predictions = torch.stack(predictions) # creating a tensor from list of tensors
    avg_predictions = torch.mean(stacked_predictions, dim=0) # calculating average (redusing number of dimensions of tensor by 1)
    labels = torch.argmax(avg_predictions, dim=1)  # finding maximal probability in every row of size 10

    submission = pd.DataFrame(
        {
            'ImageId': list(range(1, test_items + 1)),
            'Label': labels
        }
    )

    submission.to_csv('submission.csv', index=False)
