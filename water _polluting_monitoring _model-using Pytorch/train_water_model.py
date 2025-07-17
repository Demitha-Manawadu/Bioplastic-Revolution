import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Load CSV (4 columns only)
# -------------------------
df = pd.read_csv("water_sensor_labeled.csv")
df = df[['ph', 'Solids', 'Turbidity', 'Potability']].dropna()

# Round values to reduce floating-point grouping errors
df['ph_r'] = df['ph'].round(1)
df['Solids_r'] = df['Solids'].round(0)
df['Turbidity_r'] = df['Turbidity'].round(1)

# -------------------------
# Assign 4-class labels
# -------------------------
pollution_labels = []

grouped = df.groupby(['ph_r', 'Solids_r', 'Turbidity_r'])
for _, group in grouped:
    potability_vals = group['Potability'].values
    total = len(potability_vals)
    ones = np.sum(potability_vals)
    zeros = total - ones

    if ones == total:
        label = 0  # All 1 → Suitable
    elif zeros == total:
        label = 3  # All 0 → Highly polluted
    else:
        label = 2  # Mixed → Moderately polluted

    pollution_labels.extend([label] * len(group))

# Assign labels
df = df.sort_values(by=['ph_r', 'Solids_r', 'Turbidity_r']).reset_index(drop=True)
df['PollutionLevel'] = pollution_labels

# -------------------------
# Prepare final dataset
# -------------------------
df = df[['ph', 'Solids', 'Turbidity', 'PollutionLevel']]
X = df[['ph', 'Solids', 'Turbidity']].values
y = df['PollutionLevel'].values

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# PyTorch Dataset
# -------------------------
class WaterDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(WaterDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(WaterDataset(X_test, y_test), batch_size=32)

# -------------------------
# Neural Network Model
# -------------------------
class WaterClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)  # 4 pollution classes
        )

    def forward(self, x):
        return self.fc(x)

# -------------------------
# Training Loop
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WaterClassifier().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"Training on: {device}")

for epoch in range(200):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/20 - Loss: {running_loss:.4f}")

# -------------------------
# Evaluation
# -------------------------
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

print(f"\nTest Accuracy: {correct / total:.2%}")

# -------------------------
# Confusion Matrix & Report (Fixed)
# -------------------------
cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
labels = ["Suitable", "Slightly Polluted", "Moderately Polluted", "Highly Polluted"]

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Print classification report (force 4 classes)
print("\nClassification Report:")
print(classification_report(
    all_labels,
    all_preds,
    labels=[0, 1, 2, 3],
    target_names=labels,
    zero_division=0
))
torch.save(model.state_dict(), "water_model.pt")
print("Model saved as water_model.pt")
