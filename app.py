from flask import Flask, render_template, request
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch import nn

# 1. Define model
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

# 2. Load model
model = WaterClassifier()
model.load_state_dict(torch.load("water_model.pt"))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. Prepare scaler
df = pd.read_csv("water_sensor_labeled.csv")[['ph', 'Solids', 'Turbidity']].dropna()
scaler = StandardScaler()
scaler.fit(df)

# 4. Labels
labels = ["✅ Suitable", "⚠️ Slightly Polluted", "⚠️ Moderately Polluted", "❌ Highly Polluted"]

# 5. Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            ph = float(request.form["ph"])
            solids = float(request.form["solids"])
            turbidity = float(request.form["turbidity"])

            input_data = np.array([[ph, solids, turbidity]])
            input_scaled = scaler.transform(input_data)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                pred_class = torch.argmax(output, dim=1).item()
                prediction = labels[pred_class]

        except ValueError:
            prediction = "❌ Invalid input. Please enter valid numbers."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
