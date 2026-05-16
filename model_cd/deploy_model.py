import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
from clearml import Task


task = Task.init(
    project_name="Neural-Network-CICD",
    task_name="Deploy Model to Staging"
)

logger = task.get_logger()

MODEL_PATH = Path("model_package/approved_model.pt")
METADATA_PATH = Path("model_package/approved_model_metadata.json")

DEPLOY_DIR = Path("artifacts/deployed_model")
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = DEPLOY_DIR / "deployment_manifest.json"
DEPLOY_PLOT_PATH = DEPLOY_DIR / "deployment_status_plot.png"

if not MODEL_PATH.exists():
    raise FileNotFoundError("approved_model.pt not found. Run validation first.")

if not METADATA_PATH.exists():
    raise FileNotFoundError("approved_model_metadata.json not found. Run validation first.")

deployed_model_path = DEPLOY_DIR / "served_model.pt"
deployed_metadata_path = DEPLOY_DIR / "served_model_metadata.json"

shutil.copy(MODEL_PATH, deployed_model_path)
shutil.copy(METADATA_PATH, deployed_metadata_path)

model_size_mb = deployed_model_path.stat().st_size / (1024 * 1024)

deployment_start = time.time()

# Start Flask app in background
process = subprocess.Popen(
    [sys.executable, "app/serve_model.py"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

time.sleep(3)

deployment_latency_seconds = time.time() - deployment_start

deployment_manifest = {
    "deployment_stage": "staging",
    "status": "deployed",
    "deployed_at": datetime.now().isoformat(),
    "model_file": str(deployed_model_path),
    "metadata_file": str(deployed_metadata_path),
    "endpoint": "http://127.0.0.1:5000/predict",
    "health_endpoint": "http://127.0.0.1:5000/health",
    "model_size_mb": model_size_mb,
    "deployment_latency_seconds": deployment_latency_seconds,
    "flask_process_id": process.pid
}

with open(MANIFEST_PATH, "w") as f:
    json.dump(deployment_manifest, f, indent=4)

# =========================
# ClearML Scalars
# =========================
logger.report_scalar(
    title="Deployment Status",
    series="deployment_success",
    value=1,
    iteration=0
)

logger.report_scalar(
    title="Deployment Status",
    series="model_size_mb",
    value=model_size_mb,
    iteration=0
)

logger.report_scalar(
    title="Deployment Status",
    series="deployment_latency_seconds",
    value=deployment_latency_seconds,
    iteration=0
)

# =========================
# Deployment Plot
# =========================
plot_metrics = {
    "deployment_success": 1,
    "model_size_mb": model_size_mb,
    "deployment_latency_seconds": deployment_latency_seconds
}

plt.figure(figsize=(8, 5))
plt.bar(plot_metrics.keys(), plot_metrics.values())
plt.title("Deployment Summary")
plt.ylabel("Value")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(DEPLOY_PLOT_PATH)
plt.close()

logger.report_image(
    title="Deployment Plot",
    series="Deployment Summary",
    local_path=str(DEPLOY_PLOT_PATH),
    iteration=0
)

# =========================
# ClearML Artifacts
# =========================
task.upload_artifact(
    name="deployment_manifest",
    artifact_object=str(MANIFEST_PATH)
)

task.upload_artifact(
    name="served_model",
    artifact_object=str(deployed_model_path)
)

task.upload_artifact(
    name="served_model_metadata",
    artifact_object=str(deployed_metadata_path)
)

task.upload_artifact(
    name="deployment_status_plot",
    artifact_object=str(DEPLOY_PLOT_PATH)
)

logger.report_text("Model deployed to staging.")
logger.report_text("Flask inference app started in background.")
logger.report_text(json.dumps(deployment_manifest, indent=4))

print("Model deployed to staging.")
print("Flask inference app started in background.")
print(json.dumps(deployment_manifest, indent=4))

task.close()