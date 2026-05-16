import subprocess
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from clearml import Task


# =========================
# ClearML Task
# =========================
task = Task.init(
    project_name="Neural-Network-CICD",
    task_name="Check Retrain Trigger"
)

logger = task.get_logger()

# =========================
# Paths
# =========================
TRIGGER_FILE = Path("feedback/new_data_trigger.json")
TRAINING_SCRIPT = "training/train_local_model.py"

OUTPUT_DIR = Path("artifacts/retrain_trigger")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RETRAIN_LOG_PATH = OUTPUT_DIR / "retrain_log.json"
RETRAIN_PLOT_PATH = OUTPUT_DIR / "retrain_decision.png"

# Optional files created by your training script
TRAINING_METRICS_PATH = Path("artifacts/training_metrics.json")
TRAINING_HISTORY_PATH = Path("artifacts/training_history.csv")

# =========================
# Log config to ClearML
# =========================
task.connect({
    "trigger_file": str(TRIGGER_FILE),
    "training_script": TRAINING_SCRIPT,
    "training_metrics_path": str(TRAINING_METRICS_PATH),
    "training_history_path": str(TRAINING_HISTORY_PATH),
})


def log_retrain_decision(retrain: bool, reason: str, trigger_info=None):
    """
    Logs retrain decision as:
    - ClearML scalar
    - ClearML artifact
    - Local JSON
    - Simple decision plot
    """

    decision_value = 1 if retrain else 0

    decision_payload = {
        "timestamp": datetime.now().isoformat(),
        "retrain": retrain,
        "decision_value": decision_value,
        "reason": reason,
        "trigger_info": trigger_info or {}
    }

    # Save JSON locally
    with open(RETRAIN_LOG_PATH, "w") as f:
        json.dump(decision_payload, f, indent=4)

    # Upload decision artifact
    task.upload_artifact(
        name="retrain_decision",
        artifact_object=decision_payload
    )

    # Log scalar to ClearML
    logger.report_scalar(
        title="Retraining Decision",
        series="retrain_triggered",
        value=decision_value,
        iteration=0
    )

    # Log text
    logger.report_text(
        f"Retrain decision: {retrain} | Reason: {reason}"
    )

    # Create simple decision plot
    plt.figure(figsize=(5, 4))
    plt.bar(["Retrain Trigger"], [decision_value])
    plt.ylim(0, 1.2)
    plt.ylabel("Decision")
    plt.title("Retraining Trigger Decision")
    plt.text(
        0,
        decision_value + 0.05,
        "Triggered" if retrain else "Skipped",
        ha="center"
    )
    plt.tight_layout()
    plt.savefig(RETRAIN_PLOT_PATH)
    plt.close()

    # Report plot to ClearML
    logger.report_image(
        title="Retraining Decision Plot",
        series="Decision",
        local_path=str(RETRAIN_PLOT_PATH),
        iteration=0
    )

    return decision_payload


def log_training_outputs():
    """
    Logs metrics and training curves if your training script creates them.
    This makes the retraining task look richer in ClearML.
    """

    # Log metrics JSON
    if TRAINING_METRICS_PATH.exists():
        with open(TRAINING_METRICS_PATH, "r") as f:
            metrics = json.load(f)

        task.upload_artifact(
            name="training_metrics",
            artifact_object=metrics
        )

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                logger.report_scalar(
                    title="Retraining Metrics",
                    series=metric_name,
                    value=metric_value,
                    iteration=0
                )

        logger.report_text(f"Training metrics logged: {metrics}")

    else:
        logger.report_text("No training_metrics.json found. Skipping metric logging.")

    # Log training history CSV
    if TRAINING_HISTORY_PATH.exists():
        history_df = pd.read_csv(TRAINING_HISTORY_PATH)

        task.upload_artifact(
            name="training_history",
            artifact_object=history_df
        )

        # Expected columns example:
        # epoch, train_loss, val_loss, accuracy
        if "epoch" in history_df.columns:
            for col in history_df.columns:
                if col != "epoch" and pd.api.types.is_numeric_dtype(history_df[col]):
                    for _, row in history_df.iterrows():
                        logger.report_scalar(
                            title="Training Curves",
                            series=col,
                            value=float(row[col]),
                            iteration=int(row["epoch"])
                        )

            # Plot all numeric curves
            numeric_cols = [
                col for col in history_df.columns
                if col != "epoch" and pd.api.types.is_numeric_dtype(history_df[col])
            ]

            if numeric_cols:
                plt.figure(figsize=(8, 5))
                for col in numeric_cols:
                    plt.plot(history_df["epoch"], history_df[col], label=col)

                plt.xlabel("Epoch")
                plt.ylabel("Value")
                plt.title("Retraining Curves")
                plt.legend()
                plt.tight_layout()

                curve_path = OUTPUT_DIR / "training_curves.png"
                plt.savefig(curve_path)
                plt.close()

                logger.report_image(
                    title="Training Curves Plot",
                    series="Retraining",
                    local_path=str(curve_path),
                    iteration=0
                )

    else:
        logger.report_text("No training_history.csv found. Skipping curve logging.")


# =========================
# Main retrain trigger logic
# =========================
if not TRIGGER_FILE.exists():
    print("No new data detected. Retraining skipped.")

    log_retrain_decision(
        retrain=False,
        reason="No trigger file found"
    )

    task.close()
    exit(0)


with open(TRIGGER_FILE, "r") as f:
    trigger_info = json.load(f)

task.upload_artifact(
    name="trigger_info",
    artifact_object=trigger_info
)

logger.report_text(f"Trigger file found: {trigger_info}")


if trigger_info.get("new_data_received") is True:
    print("New data detected.")
    print("Retraining triggered...")

    log_retrain_decision(
        retrain=True,
        reason="New data trigger detected",
        trigger_info=trigger_info
    )

    # Run training script
    result = subprocess.run(
        ["python", TRAINING_SCRIPT],
        check=True,
        capture_output=True,
        text=True
    )

    # Log stdout/stderr into ClearML
    logger.report_text("Training stdout:")
    logger.report_text(result.stdout)

    if result.stderr:
        logger.report_text("Training stderr:")
        logger.report_text(result.stderr)

    # Log training metrics and plots
    log_training_outputs()

    # Remove trigger file after successful retraining
    TRIGGER_FILE.unlink()

    print("Retraining complete. Trigger file removed.")
    logger.report_text("Retraining complete. Trigger file removed.")

else:
    print("Trigger file exists, but retraining condition not met.")

    log_retrain_decision(
        retrain=False,
        reason="Trigger file exists but new_data_received is not True",
        trigger_info=trigger_info
    )


task.close()