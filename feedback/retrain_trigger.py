import subprocess
import json
from pathlib import Path
from clearml import Task

task = Task.init(
    project_name="Neural-Network-CICD",
    task_name="Check Retrain Trigger"
)

TRIGGER_FILE = Path("feedback/new_data_trigger.json")
TRAINING_SCRIPT = "training/train_local_cnn.py"

# Log config to ClearML
task.connect({
    "trigger_file": str(TRIGGER_FILE),
    "training_script": TRAINING_SCRIPT
})

if not TRIGGER_FILE.exists():
    print("No new data detected. Retraining skipped.")

    task.upload_artifact(
        name="retrain_decision",
        artifact_object={
            "retrain": False,
            "reason": "No trigger file found"
        }
    )

    exit(0)

with open(TRIGGER_FILE, "r") as f:
    trigger_info = json.load(f)

task.upload_artifact(
    name="trigger_info",
    artifact_object=trigger_info
)

if trigger_info.get("new_data_received") is True:
    print("New data detected.")
    print("Retraining triggered...")

    task.upload_artifact(
        name="retrain_decision",
        artifact_object={
            "retrain": True,
            "reason": "New data trigger detected"
        }
    )

    subprocess.run(
        ["python", TRAINING_SCRIPT],
        check=True
    )

    TRIGGER_FILE.unlink()
    print("Retraining complete. Trigger file removed.")

else:
    print("Trigger file exists, but retraining condition not met.")

    task.upload_artifact(
        name="retrain_decision",
        artifact_object={
            "retrain": False,
            "reason": "Trigger file exists but condition not met"
        }
    )