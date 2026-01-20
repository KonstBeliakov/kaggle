# https://www.kaggle.com/competitions/binary-classification-in-space

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X = train_df.drop('target', axis=1)
y = train_df['target']

counter = sum(y)
print(f"Total number of galaxies in training: {len(y)}")
print(f"Number of dwarf galaxies: {counter}")
print(f"Number of common galaxies: {len(y) - counter}") # we need to somehow balance the dataset


X_test_final = test_df.drop('Id', axis=1)
test_ids = test_df['Id']

X = X.fillna(X.median()) # handling null values in the dataset
X_test_final = X_test_final.fillna(X_test_final.median())

# Scaling features (Neural Networks perform better with scaled data)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_test_scaled = scaler.transform(X_test_final)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_final.values, dtype=torch.float32)


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.hidden_size = 64
        self.network = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()  # Giving probabilities of two classes
        )

    def forward(self, x):
        return self.network(x)


model = MLP(X_train.shape[1])

# This approach makes 90% predictions as dwarf galaxies
# num_pos = y_train.sum()
# num_neg = len(y_train) - num_pos
# pos_weight = torch.tensor([num_neg / num_pos])
#
# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_classes = (predictions > 0.5).int().numpy() # from probability to 0/1

classes = predicted_classes.flatten()
submission = pd.DataFrame({
    'target': classes,
    'Id': test_ids,
})

submission.to_csv('submission.csv', index=False)
print("Predictions saved to submission.csv")
counter = sum(classes)
print(f"Dwarf galaxies predicted: {len(classes) - counter}, common galaxies predicted: {counter}")
