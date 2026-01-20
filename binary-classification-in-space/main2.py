# https://www.kaggle.com/competitions/binary-classification-in-space

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X = train_df.drop('target', axis=1)
y = train_df['target']
X_test_final = test_df.drop('Id', axis=1)
test_ids = test_df['Id']

X = X.fillna(X.median())
X_test_final = X_test_final.fillna(X_test_final.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test_final)


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.hidden_size = 128
        self.network = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# Cross-Validation
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
all_test_predictions = []

for fold, (train_index, val_index) in enumerate(kf.split(X_scaled)):
    print(f"Training Fold {fold + 1}...")

    # Split data for this fold
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    model = MLP(X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    epochs = 150
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)

        test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        fold_predictions = model(test_tensor).numpy()
        all_test_predictions.append(fold_predictions)

    print(f'Fold {fold + 1} finished. Val Loss: {val_loss.item():.4f}')

# Average predictions from all folds (Ensemble)
final_predictions = np.mean(all_test_predictions, axis=0)
predicted_classes = (final_predictions > 0.93).astype(int)

classes = predicted_classes.flatten()
submission = pd.DataFrame({
    'target': classes,
    'Id': test_ids,
})

common_galaxies = sum(classes)
submission.to_csv('submission.csv', index=False)
print(f"Predictions saved to submission.csv using {n_splits}-fold CV ensembling.")
print(f"Dwarf galaxies predicted: {len(classes) - common_galaxies}, common galaxies predicted: {common_galaxies}")
