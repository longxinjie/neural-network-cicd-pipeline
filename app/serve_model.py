import json
from pathlib import Path

import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)

MODEL_PATH = Path("artifacts/deployed_model/served_model.pt")
METADATA_PATH = Path("artifacts/deployed_model/served_model_metadata.json")


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Deployed model not found at: {MODEL_PATH}"
    )

checkpoint = torch.load(MODEL_PATH, map_location=device)

model = SimpleCNN().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "model_architecture": metadata.get("model_architecture"),
        "accuracy": metadata.get("accuracy")
    })


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({
            "error": "No image file uploaded"
        }), 400

    file = request.files["file"]

    image_bytes = file.read()

    image = Image.open(io.BytesIO(image_bytes))

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(probabilities, 1)

    return jsonify({
        "predicted_digit": int(prediction.item()),
        "confidence": round(float(confidence.item()), 4),
        "model_metadata": metadata
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )