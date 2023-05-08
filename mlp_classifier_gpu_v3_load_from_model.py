import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

device = torch.device('cuda:0')
# device = torch.device('cpu')

MODEL_ROOT = 'model/'

# Load the data
with open(MODEL_ROOT + 'X_resampled.pkl', 'rb') as f:
    X_resampled = pickle.load(f)

with open(MODEL_ROOT + 'y_resampled.pkl', 'rb') as f:
    y_resampled = pickle.load(f)

with open(MODEL_ROOT + 'X.pkl', 'rb') as f:
    X = pickle.load(f)

with open(MODEL_ROOT + 'X_test_scaled.pkl', 'rb') as f:
    X_test_scaled = pickle.load(f)

with open(MODEL_ROOT + 'y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)

with open(MODEL_ROOT + 'y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)


class MLPClassifierPytorch(nn.Module):
    def __init__(self):
        super(MLPClassifierPytorch, self).__init__()
        self.layer1 = nn.Linear(X.shape[1], 50)
        self.layer2 = nn.Linear(50, 30)
        self.layer3 = nn.Linear(30, 2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


mlp = MLPClassifierPytorch().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp.parameters(), lr=0.001)

X_train_tensor = torch.FloatTensor(X_resampled).to(device)
y_train_tensor = torch.LongTensor(y_resampled).to(device)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = mlp(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

# Save the trained model
torch.save(mlp.state_dict(), MODEL_ROOT + "mlp_model.pth")
print("Model saved as mlp_model.pth")

X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_test_tensor = torch.LongTensor(y_test.values).to(device)

with torch.no_grad():
    test_outputs = mlp(X_test_tensor)
    _, y_pred = torch.max(test_outputs, 1)

accuracy = accuracy_score(y_test, y_pred.cpu().numpy())
print("Accuracy of the MLP Classifier with SMOTE (PyTorch): {:.2f}%".format(accuracy * 100))
