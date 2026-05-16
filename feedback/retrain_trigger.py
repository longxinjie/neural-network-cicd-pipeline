import subprocess
import json
from pathlib import Path

TRIGGER_FILE = Path("feedback/new_data_trigger.json")
TRAINING_SCRIPT = "training/train_local_cnn.py"

if not TRIGGER_FILE.exists():
    print("No new data detected. Retraining skipped.")
    exit(0)

with open(TRIGGER_FILE, "r") as f:
    trigger_info = json.load(f)

if trigger_info.get("new_data_received") is True:
    print("New data detected.")
    print("Retraining triggered...")

    subprocess.run(
        ["python", TRAINING_SCRIPT],
        check=True
    )

    TRIGGER_FILE.unlink()
    print("Retraining complete. Trigger file removed.")

else:
    print("Trigger file exists, but retraining condition not met.")