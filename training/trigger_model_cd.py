import subprocess
from pathlib import Path
from clearml import Task

task = Task.init(
    project_name="Neural-Network-CICD",
    task_name="Trigger Model CD Pipeline"
)

handoff_file = Path("artifacts/approved_checkpoint.json")

if not handoff_file.exists():
    print("No approved checkpoint found yet. This is expected if running this task alone.")
    print("Task registered successfully. Model CD will only run after validation creates the handoff file.")
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

subprocess.Popen(
    ["python", "model_cd/clearml_model_cd_pipeline.py"]
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

print("Model CD pipeline triggered.")