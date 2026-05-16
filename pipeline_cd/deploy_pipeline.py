import json
import shutil
from pathlib import Path
from datetime import datetime

SOURCE_APP = Path("app/serve_model.py")
SOURCE_MONITORING = Path("monitoring/monitor_model.py")
SOURCE_MODEL_CD = Path("model_cd")

PIPELINE_DEPLOY_DIR = Path("artifacts/pipeline_staging")
PIPELINE_DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

if not SOURCE_APP.exists():
    raise FileNotFoundError("app/serve_model.py not found")

if not SOURCE_MONITORING.exists():
    raise FileNotFoundError("monitoring/monitor_model.py not found")

if not SOURCE_MODEL_CD.exists():
    raise FileNotFoundError("model_cd folder not found")

# Deploy serving app
shutil.copy(SOURCE_APP, PIPELINE_DEPLOY_DIR / "serve_model.py")

# Deploy monitoring script
shutil.copy(SOURCE_MONITORING, PIPELINE_DEPLOY_DIR / "monitor_model.py")

# Deploy model CD workflow scripts
model_cd_target = PIPELINE_DEPLOY_DIR / "model_cd"
if model_cd_target.exists():
    shutil.rmtree(model_cd_target)

shutil.copytree(SOURCE_MODEL_CD, model_cd_target)

manifest = {
    "deployment_type": "pipeline_cd",
    "stage": "staging",
    "status": "deployed",
    "deployed_at": datetime.now().isoformat(),
    "deployed_components": [
        "Flask serving application",
        "Model CD workflow scripts",
        "Monitoring script"
    ],
    "purpose": "Deploy ML pipeline infrastructure and serving workflow, not model weights"
}

with open(PIPELINE_DEPLOY_DIR / "pipeline_deployment_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("Pipeline CD completed.")
print(json.dumps(manifest, indent=2))