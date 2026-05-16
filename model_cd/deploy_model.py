import json
import shutil
from pathlib import Path
from datetime import datetime

MODEL_PATH = Path("model_package/approved_model.pt")
METADATA_PATH = Path("model_package/approved_model_metadata.json")

DEPLOY_DIR = Path("artifacts/deployed_model")
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError("approved_model.pt not found. Run validation first.")

if not METADATA_PATH.exists():
    raise FileNotFoundError("approved_model_metadata.json not found. Run validation first.")

# Copy approved model into deployment folder
deployed_model_path = DEPLOY_DIR / "served_model.pt"
deployed_metadata_path = DEPLOY_DIR / "served_model_metadata.json"

shutil.copy(MODEL_PATH, deployed_model_path)
shutil.copy(METADATA_PATH, deployed_metadata_path)

deployment_manifest = {
    "deployment_stage": "staging",
    "status": "deployed",
    "deployed_at": datetime.now().isoformat(),
    "model_file": str(deployed_model_path),
    "metadata_file": str(deployed_metadata_path)
}

with open(DEPLOY_DIR / "deployment_manifest.json", "w") as f:
    json.dump(deployment_manifest, f, indent=2)

print("Model deployed to staging.")
print(json.dumps(deployment_manifest, indent=2))