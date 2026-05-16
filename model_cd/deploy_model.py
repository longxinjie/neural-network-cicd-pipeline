import json
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime

from clearml import Task

task = Task.init(
    project_name="Neural-Network-CICD",
    task_name="Deploy Model to Staging"
)

MODEL_PATH = Path("model_package/approved_model.pt")
METADATA_PATH = Path("model_package/approved_model_metadata.json")

DEPLOY_DIR = Path("artifacts/deployed_model")
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError("approved_model.pt not found. Run validation first.")

if not METADATA_PATH.exists():
    raise FileNotFoundError("approved_model_metadata.json not found. Run validation first.")

deployed_model_path = DEPLOY_DIR / "served_model.pt"
deployed_metadata_path = DEPLOY_DIR / "served_model_metadata.json"

shutil.copy(MODEL_PATH, deployed_model_path)
shutil.copy(METADATA_PATH, deployed_metadata_path)

deployment_manifest = {
    "deployment_stage": "staging",
    "status": "deployed",
    "deployed_at": datetime.now().isoformat(),
    "model_file": str(deployed_model_path),
    "metadata_file": str(deployed_metadata_path),
    "endpoint": "http://127.0.0.1:5000/predict",
    "health_endpoint": "http://127.0.0.1:5000/health"
}

with open(DEPLOY_DIR / "deployment_manifest.json", "w") as f:
    json.dump(deployment_manifest, f, indent=2)

# Start Flask app in background
subprocess.Popen(
    [sys.executable, "app/serve_model.py"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

task.upload_artifact(
    name="deployment_manifest",
    artifact_object="artifacts/deployed_model/deployment_manifest.json"
)

task.upload_artifact(
    name="served_model",
    artifact_object=str(deployed_model_path)
)

print("Model deployed to staging.")
print("Flask inference app started in background.")
print(json.dumps(deployment_manifest, indent=2))