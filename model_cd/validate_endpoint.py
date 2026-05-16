import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import requests
from PIL import Image, ImageDraw

from clearml import Task


task = Task.init(
    project_name="Neural-Network-CICD",
    task_name="Validate Deployment Endpoint"
)

logger = task.get_logger()

OUTPUT_DIR = Path("reports")
FIGURE_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "http://127.0.0.1:5000"

RESULT_PATH = OUTPUT_DIR / "endpoint_validation_result.json"
LATENCY_PLOT_PATH = FIGURE_DIR / "endpoint_latency_plot.png"
VALIDATION_PLOT_PATH = FIGURE_DIR / "endpoint_validation_summary.png"
TEST_IMAGE_PATH = FIGURE_DIR / "sample_digit_for_endpoint.png"

# =========================
# 1. Health check
# =========================
health_start = time.time()

health_response = requests.get(
    f"{BASE_URL}/health",
    timeout=10
)

health_latency_ms = (time.time() - health_start) * 1000
health = health_response.json()

if health_response.status_code != 200 or health.get("status") != "healthy":
    raise ValueError(f"Health check failed: {health}")

logger.report_scalar(
    title="Endpoint Health",
    series="health_check_passed",
    value=1,
    iteration=0
)

logger.report_scalar(
    title="Endpoint Latency",
    series="health_latency_ms",
    value=health_latency_ms,
    iteration=0
)

# =========================
# 2. Create test digit image
# =========================
img = Image.new("L", (28, 28), color=0)
draw = ImageDraw.Draw(img)
draw.text((9, 4), "7", fill=255)
img.save(TEST_IMAGE_PATH)

logger.report_image(
    title="Endpoint Test Input",
    series="sample_digit",
    local_path=str(TEST_IMAGE_PATH),
    iteration=0
)

# =========================
# 3. Send prediction request
# =========================
predict_start = time.time()

with open(TEST_IMAGE_PATH, "rb") as f:
    response = requests.post(
        f"{BASE_URL}/predict",
        files={"file": f},
        timeout=10
    )

prediction_latency_ms = (time.time() - predict_start) * 1000

try:
    prediction = response.json()
except Exception:
    raise ValueError(f"Prediction endpoint did not return JSON: {response.text}")

if response.status_code != 200:
    raise ValueError(f"Prediction failed: {prediction}")

predicted_class = prediction.get("prediction", None)
confidence = prediction.get("confidence", None)

if confidence is None:
    confidence = prediction.get("probability", 0)

# =========================
# 4. Log prediction scalars
# =========================
logger.report_scalar(
    title="Endpoint Validation",
    series="prediction_request_passed",
    value=1,
    iteration=0
)

logger.report_scalar(
    title="Endpoint Latency",
    series="prediction_latency_ms",
    value=prediction_latency_ms,
    iteration=0
)

if isinstance(predicted_class, (int, float)):
    logger.report_scalar(
        title="Endpoint Prediction",
        series="predicted_class",
        value=float(predicted_class),
        iteration=0
    )

if isinstance(confidence, (int, float)):
    logger.report_scalar(
        title="Endpoint Prediction",
        series="prediction_confidence",
        value=float(confidence),
        iteration=0
    )

# =========================
# 5. Create plots
# =========================
latency_metrics = {
    "health_latency_ms": health_latency_ms,
    "prediction_latency_ms": prediction_latency_ms
}

plt.figure(figsize=(7, 5))
plt.bar(latency_metrics.keys(), latency_metrics.values())
plt.title("Endpoint Latency")
plt.ylabel("Milliseconds")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(LATENCY_PLOT_PATH)
plt.close()

logger.report_image(
    title="Endpoint Latency Plot",
    series="Latency",
    local_path=str(LATENCY_PLOT_PATH),
    iteration=0
)

validation_metrics = {
    "health_check_passed": 1,
    "prediction_request_passed": 1
}

plt.figure(figsize=(7, 5))
plt.bar(validation_metrics.keys(), validation_metrics.values())
plt.ylim(0, 1.2)
plt.title("Endpoint Validation Summary")
plt.ylabel("Pass / Fail")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(VALIDATION_PLOT_PATH)
plt.close()

logger.report_image(
    title="Endpoint Validation Plot",
    series="Validation Summary",
    local_path=str(VALIDATION_PLOT_PATH),
    iteration=0
)

# =========================
# 6. Save result artifact
# =========================
result = {
    "endpoint": BASE_URL,
    "health_check": health,
    "prediction_response": prediction,
    "status": "PASSED",
    "health_latency_ms": health_latency_ms,
    "prediction_latency_ms": prediction_latency_ms,
    "predicted_class": predicted_class,
    "prediction_confidence": confidence
}

with open(RESULT_PATH, "w") as f:
    json.dump(result, f, indent=4)

task.upload_artifact(
    name="endpoint_validation_result",
    artifact_object=str(RESULT_PATH)
)

task.upload_artifact(
    name="sample_digit_for_endpoint",
    artifact_object=str(TEST_IMAGE_PATH)
)

task.upload_artifact(
    name="endpoint_latency_plot",
    artifact_object=str(LATENCY_PLOT_PATH)
)

task.upload_artifact(
    name="endpoint_validation_plot",
    artifact_object=str(VALIDATION_PLOT_PATH)
)

logger.report_text("Endpoint validation passed.")
logger.report_text(json.dumps(result, indent=4))

print("Endpoint validation passed.")
print(json.dumps(result, indent=4))

task.close()