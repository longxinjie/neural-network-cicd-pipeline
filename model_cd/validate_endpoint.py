import json
from pathlib import Path
import requests
from PIL import Image, ImageDraw

from clearml import Task

task = Task.init(
    project_name="Neural-Network-CICD",
    task_name="Validate Deployment Endpoint"
)

OUTPUT_DIR = Path("reports")
FIGURE_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "http://127.0.0.1:5000"

# 1. Health check
health = requests.get(f"{BASE_URL}/health", timeout=10).json()

if health.get("status") != "healthy":
    raise ValueError("Health check failed")

# 2. Create a simple test digit image
test_image_path = FIGURE_DIR / "sample_digit_for_endpoint.png"

img = Image.new("L", (28, 28), color=0)
draw = ImageDraw.Draw(img)
draw.text((9, 4), "7", fill=255)
img.save(test_image_path)

# 3. Send image to prediction endpoint
with open(test_image_path, "rb") as f:
    response = requests.post(
        f"{BASE_URL}/predict",
        files={"file": f},
        timeout=10
    )

prediction = response.json()

if response.status_code != 200:
    raise ValueError(f"Prediction failed: {prediction}")

result = {
    "endpoint": BASE_URL,
    "health_check": health,
    "prediction_response": prediction,
    "status": "PASSED"
}

with open(OUTPUT_DIR / "endpoint_validation_result.json", "w") as f:
    json.dump(result, f, indent=2)

task.upload_artifact(
    name="endpoint_validation_result",
    artifact_object="reports/endpoint_validation_result.json"
)

task.upload_artifact(
    name="sample_digit_for_endpoint",
    artifact_object=str(test_image_path)
)

print("Endpoint validation passed.")
print(json.dumps(result, indent=2))