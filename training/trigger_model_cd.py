import os
import sys
import subprocess
from pathlib import Path
from clearml import Task

task = Task.init(
    project_name="Neural-Network-CICD",
    task_name="Trigger Model CD Pipeline"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
handoff_file = PROJECT_ROOT / "artifacts" / "approved_checkpoint.json"

if not handoff_file.exists():
    print("No approved checkpoint found yet. This is expected if running this task alone.")
    task.upload_artifact(
        name="trigger_status",
        artifact_object={
            "model_cd_triggered": False,
            "reason": "No approved checkpoint found"
        }
    )
    exit(0)

print("Approved checkpoint found.")
print("Triggering Model CD pipeline...")

env = os.environ.copy()

# Important: prevent the child process from attaching to this ClearML task
env.pop("CLEARML_TASK_ID", None)
env.pop("TRAINS_TASK_ID", None)

subprocess.Popen(
    [sys.executable, "model_cd/clearml_model_cd_pipeline.py"],
    cwd=PROJECT_ROOT,
    env=env
)

task.upload_artifact(
    name="approved_checkpoint_handoff",
    artifact_object=str(handoff_file)
)

task.upload_artifact(
    name="trigger_status",
    artifact_object={
        "model_cd_triggered": True,
        "reason": "Approved checkpoint found"
    }
)

print("Model CD pipeline triggered as a separate process.")