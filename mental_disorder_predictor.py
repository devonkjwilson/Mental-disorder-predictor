import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# === Load and preprocess sample data ===
# Your CSV must include these columns at minimum
# age, gender, family_history_anxiety, family_history_depression, family_history_bipolar,
# target_anxiety, target_depression, target_bipolar

df = pd.read_csv("mental_health_dataset.csv")

# Encode gender
df["gender"] = LabelEncoder().fit_transform(df["gender"])

# Features and targets
X = df[["age", "gender", "family_history_anxiety", "family_history_depression", "family_history_bipolar"]].values
y = df[["target_anxiety", "target_depression", "target_bipolar"]].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# === Define Neural Network ===
class DisorderRiskModel(nn.Module):
    def __init__(self):
        super(DisorderRiskModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),      # 3 outputs: anxiety, depression, bipolar
            nn.Sigmoid()          # Outputs as probabilities (0–1)
        )

    def forward(self, x):
        return self.model(x)

model = DisorderRiskModel()

# === Training Config ===
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 200

# === Training Loop ===
for epoch in range(epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# === Evaluation ===
model.eval()
predictions = model(X_test_tensor).detach().numpy()
print("Sample Predictions:\n", predictions[:5])
