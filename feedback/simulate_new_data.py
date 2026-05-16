from pathlib import Path
from datetime import datetime
import json

from clearml import Task

task = Task.init(
    project_name="Neural-Network-CICD",
    task_name="Simulate New Data"
)

FEEDBACK_DIR = Path("feedback")
FEEDBACK_DIR.mkdir(exist_ok=True)

trigger_file = FEEDBACK_DIR / "new_data_trigger.json"

trigger_info = {
    "new_data_received": True,
    "source": "simulated_new_mnist_data",
    "created_at": datetime.now().isoformat(),
    "reason": "New production-like data received for retraining demo"
}

with open(trigger_file, "w") as f:
    json.dump(trigger_info, f, indent=2)

task.upload_artifact(
    name="new_data_trigger",
    artifact_object=trigger_info
)

print("Simulated new data arrival.")
print(f"Trigger file created at: {trigger_file}")
print(trigger_info)