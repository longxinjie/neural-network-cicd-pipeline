from pathlib import Path
from datetime import datetime
import json

import matplotlib.pyplot as plt
import pandas as pd

from clearml import Task


# =========================
# ClearML Task
# =========================
task = Task.init(
    project_name="Neural-Network-CICD",
    task_name="Simulate New Data"
)

logger = task.get_logger()

# =========================
# Paths
# =========================
FEEDBACK_DIR = Path("feedback")
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = Path("artifacts/feedback_loop")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRIGGER_FILE = FEEDBACK_DIR / "new_data_trigger.json"

PLOT_PATH = OUTPUT_DIR / "new_data_plot.png"
CSV_PATH = OUTPUT_DIR / "new_data_summary.csv"

# =========================
# Simulated feedback info
# =========================
trigger_info = {
    "new_data_received": True,
    "source": "simulated_new_mnist_data",
    "created_at": datetime.now().isoformat(),
    "reason": "New production-like data received for retraining demo",
    "num_new_samples": 100,
    "drift_score": 0.42,
    "avg_prediction_confidence": 0.71,
    "trigger_stage": "feedback_loop"
}

# =========================
# Save trigger file
# =========================
with open(TRIGGER_FILE, "w") as f:
    json.dump(trigger_info, f, indent=4)

# =========================
# Upload trigger artifact
# =========================
task.upload_artifact(
    name="new_data_trigger",
    artifact_object=trigger_info
)

# =========================
# Create fake monitoring dataframe
# =========================
monitor_df = pd.DataFrame({
    "metric": [
        "num_new_samples",
        "drift_score",
        "avg_prediction_confidence"
    ],
    "value": [
        trigger_info["num_new_samples"],
        trigger_info["drift_score"],
        trigger_info["avg_prediction_confidence"]
    ]
})

monitor_df.to_csv(CSV_PATH, index=False)

task.upload_artifact(
    name="feedback_metrics",
    artifact_object=monitor_df
)

# =========================
# Log scalars to ClearML
# =========================
logger.report_scalar(
    title="Feedback Loop",
    series="new_data_received",
    value=1,
    iteration=0
)

logger.report_scalar(
    title="Feedback Loop",
    series="num_new_samples",
    value=trigger_info["num_new_samples"],
    iteration=0
)

logger.report_scalar(
    title="Feedback Loop",
    series="drift_score",
    value=trigger_info["drift_score"],
    iteration=0
)

logger.report_scalar(
    title="Feedback Loop",
    series="avg_prediction_confidence",
    value=trigger_info["avg_prediction_confidence"],
    iteration=0
)

# =========================
# Generate monitoring plot
# =========================
plt.figure(figsize=(8, 5))

plt.bar(
    monitor_df["metric"],
    monitor_df["value"]
)

plt.title("Feedback Loop Monitoring Metrics")
plt.ylabel("Value")
plt.xticks(rotation=10)

plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.close()

# =========================
# Upload plot to ClearML
# =========================
logger.report_image(
    title="Feedback Monitoring Plot",
    series="feedback_metrics",
    local_path=str(PLOT_PATH),
    iteration=0
)

# =========================
# Log text
# =========================
logger.report_text("Simulated production feedback received.")
logger.report_text(f"Trigger file created at: {TRIGGER_FILE}")
logger.report_text(str(trigger_info))

print("Simulated new data arrival.")
print(f"Trigger file created at: {TRIGGER_FILE}")
print(trigger_info)

task.close()