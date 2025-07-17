import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch import nn

# -------------------------
# Model Definition
# -------------------------
class WaterClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )
    def forward(self, x):
        return self.fc(x)

# -------------------------
# Load Model
# -------------------------
model = WaterClassifier()
model.load_state_dict(torch.load("water_model.pt"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------------
# Fit Scaler
# -------------------------
df = pd.read_csv("water_sensor_labeled.csv")[['ph', 'Solids', 'Turbidity']].dropna()
scaler = StandardScaler()
scaler.fit(df)

# -------------------------
# Label Map
# -------------------------
labels = ["✅ Suitable", "⚠️ Slightly Polluted", "⚠️ Moderately Polluted", "❌ Highly Polluted"]

# -------------------------
# User Input Loop
# -------------------------
print("💧 Water Quality Predictor (pH, TDS, Turbidity)")
print("Type 'exit' at any time to quit.\n")

while True:
    try:
        ph_input = input("Enter pH value: ")
        if ph_input.lower() == 'exit': break
        solids_input = input("Enter Solids (TDS) in mg/L: ")
        if solids_input.lower() == 'exit': break
        turbidity_input = input("Enter Turbidity (NTU): ")
        if turbidity_input.lower() == 'exit': break

        # Convert to float
        ph = float(ph_input)
        solids = float(solids_input)
        turbidity = float(turbidity_input)

        # Prepare input
        user_data = np.array([[ph, solids, turbidity]])
        user_scaled = scaler.transform(user_data)
        input_tensor = torch.tensor(user_scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        print(f"🔎 Prediction: {labels[predicted_class]} (Class {predicted_class})\n")

    except ValueError:
        print("❌ Invalid input. Please enter numeric values.\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
