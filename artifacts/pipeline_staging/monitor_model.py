import json
import random
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

REPORT_DIR = Path("reports")
FIGURE_DIR = Path("reports/figures")
MONITORING_LOG = REPORT_DIR / "monitoring_log.json"

REPORT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Simulate production monitoring metrics
new_entry = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "model_version": "served_model",
    "accuracy_estimate": round(random.uniform(0.88, 0.97), 4),
    "average_confidence": round(random.uniform(0.80, 0.99), 4),
    "prediction_volume": random.randint(80, 200),
    "drift_score": round(random.uniform(0.01, 0.18), 4),
}

if MONITORING_LOG.exists():
    with open(MONITORING_LOG, "r") as f:
        logs = json.load(f)
else:
    logs = []

logs.append(new_entry)

with open(MONITORING_LOG, "w") as f:
    json.dump(logs, f, indent=2)

df = pd.DataFrame(logs)
df["run"] = range(1, len(df) + 1)

# Accuracy trend
plt.figure(figsize=(8, 4))
plt.plot(df["run"], df["accuracy_estimate"], marker="o")
plt.axhline(y=0.80, linestyle="--")
plt.title("Model Accuracy Monitoring Trend")
plt.xlabel("Monitoring Run")
plt.ylabel("Estimated Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "monitoring_accuracy_trend.png")
plt.close()

# Confidence trend
plt.figure(figsize=(8, 4))
plt.plot(df["run"], df["average_confidence"], marker="o")
plt.title("Average Prediction Confidence Trend")
plt.xlabel("Monitoring Run")
plt.ylabel("Average Confidence")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "monitoring_confidence_trend.png")
plt.close()

# Drift trend
plt.figure(figsize=(8, 4))
plt.plot(df["run"], df["drift_score"], marker="o")
plt.axhline(y=0.15, linestyle="--")
plt.title("Simulated Drift Score Trend")
plt.xlabel("Monitoring Run")
plt.ylabel("Drift Score")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "monitoring_drift_trend.png")
plt.close()

print("Monitoring updated.")
print(json.dumps(new_entry, indent=2))
print("Generated monitoring visuals in reports/figures/")