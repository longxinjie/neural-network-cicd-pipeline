import subprocess
from pathlib import Path
from clearml import Task

task = Task.init(
    project_name="Neural-Network-CICD",
    task_name="Trigger Model CD Pipeline"
)

handoff_file = Path("artifacts/approved_checkpoint.json")

if not handoff_file.exists():
    raise FileNotFoundError("No approved checkpoint found. Model CD will not run.")

print("Approved checkpoint found.")
print("Triggering Model CD pipeline...")

subprocess.run(
    ["python", "model_cd/clearml_model_cd_pipeline.py"],
    check=True
)

task.upload_artifact(
    name="approved_checkpoint_handoff",
    artifact_object=str(handoff_file)
)

print("Model CD pipeline triggered.")