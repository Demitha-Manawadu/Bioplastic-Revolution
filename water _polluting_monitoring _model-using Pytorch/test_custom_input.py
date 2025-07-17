import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch import nn

# -------------------------
# Define model architecture
# -------------------------
class WaterClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)  # 4 classes
        )

    def forward(self, x):
        return self.fc(x)

# -------------------------
# Load model
# -------------------------
model = WaterClassifier()
model.load_state_dict(torch.load("water_model.pt"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------------
# Prepare scaler (fit on full dataset again)
# -------------------------
df = pd.read_csv("water_sensor_labeled.csv")[['ph', 'Solids', 'Turbidity']].dropna()
scaler = StandardScaler()
scaler.fit(df)

# -------------------------
# Manual Input3.7
# -------------------------
try:
    ph = float(input("Enter pH value: "))
    solids = float(input("Enter Solids (TDS) in mg/L: "))
    turbidity = float(input("Enter Turbidity (NTU): "))

    custom_input = np.array([[ph, solids, turbidity]])
    custom_scaled = scaler.transform(custom_input)
    input_tensor = torch.tensor(custom_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    labels = ["Suitable", "Slightly Polluted", "Moderately Polluted", "Highly Polluted"]
    print(f"\nPrediction: {labels[predicted_class]} (Class {predicted_class})")

except ValueError:
    print("❌ Invalid input. Please enter numeric values.")
