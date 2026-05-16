import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from clearml import Task


task = Task.init(
    project_name="Neural-Network-CICD",
    task_name="Validate Model"
)

logger = task.get_logger()

OUTPUT_DIR = Path("reports")
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_PACKAGE_DIR = Path("model_package")
ARTIFACTS_DIR = Path("artifacts")

OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PACKAGE_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

MIN_ACCURACY = 0.80


def plot_accuracy_vs_threshold(accuracy, min_accuracy, status):
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(["Model Accuracy", "Required Threshold"], [accuracy, min_accuracy])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(f"Validation Decision: {status}")

    for index, value in enumerate([accuracy, min_accuracy]):
        ax.text(
            index,
            value + 0.02,
            f"{value:.2%}",
            ha="center"
        )

    fig.tight_layout()

    path = FIGURE_DIR / "validation_accuracy_vs_threshold.png"
    fig.savefig(path)

    logger.report_matplotlib_figure(
        title="Validation Plots",
        series="Accuracy vs Threshold",
        figure=fig,
        iteration=0
    )

    plt.close(fig)
    return path


def plot_validation_gate(accuracy, min_accuracy, status):
    fig, ax = plt.subplots(figsize=(8, 2.5))

    ax.axhline(0, xmin=0, xmax=1)
    ax.scatter(accuracy, 0, s=250, label=f"Accuracy: {accuracy:.2%}")
    ax.axvline(min_accuracy, linestyle="--", label=f"Threshold: {min_accuracy:.2%}")

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Accuracy")
    ax.set_title(f"Model Promotion Gate: {status}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)

    fig.tight_layout()

    path = FIGURE_DIR / "validation_gate.png"
    fig.savefig(path)

    logger.report_matplotlib_figure(
        title="Validation Plots",
        series="Promotion Gate",
        figure=fig,
        iteration=0
    )

    plt.close(fig)
    return path


def validate():
    evaluation_path = OUTPUT_DIR / "evaluation_result.json"

    if not evaluation_path.exists():
        raise FileNotFoundError("No evaluation_result.json found. Run evaluation first.")

    with open(evaluation_path, "r") as f:
        evaluation_result = json.load(f)

    checkpoint_path = Path(evaluation_result["checkpoint"])
    accuracy = float(evaluation_result["accuracy"])

    status = "PASSED" if accuracy >= MIN_ACCURACY else "FAILED"

    validation_result = {
        "checkpoint": str(checkpoint_path),
        "accuracy": accuracy,
        "min_accuracy": MIN_ACCURACY,
        "status": status
    }

    validation_result_path = OUTPUT_DIR / "validation_result.json"

    with open(validation_result_path, "w") as f:
        json.dump(validation_result, f, indent=2)

    accuracy_threshold_plot = plot_accuracy_vs_threshold(
        accuracy=accuracy,
        min_accuracy=MIN_ACCURACY,
        status=status
    )

    validation_gate_plot = plot_validation_gate(
        accuracy=accuracy,
        min_accuracy=MIN_ACCURACY,
        status=status
    )

    logger.report_scalar(
        title="Validation Metrics",
        series="accuracy",
        value=accuracy,
        iteration=0
    )

    logger.report_scalar(
        title="Validation Metrics",
        series="min_accuracy",
        value=MIN_ACCURACY,
        iteration=0
    )

    logger.report_text(
        f"""
# Model Validation Summary

Status: {status}

Accuracy: {accuracy:.4f}  
Required Threshold: {MIN_ACCURACY:.4f}  

Checkpoint: {checkpoint_path}
"""
    )

    task.upload_artifact(
        name="validation_result",
        artifact_object=str(validation_result_path)
    )

    task.upload_artifact(
        name="validation_accuracy_vs_threshold",
        artifact_object=str(accuracy_threshold_plot)
    )

    task.upload_artifact(
        name="validation_gate",
        artifact_object=str(validation_gate_plot)
    )

    if status == "FAILED":
        raise ValueError(
            f"Model rejected. Accuracy {accuracy:.4f} < threshold {MIN_ACCURACY:.4f}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    approved_path = MODEL_PACKAGE_DIR / "approved_model.pt"
    approved_metadata_path = MODEL_PACKAGE_DIR / "approved_model_metadata.json"

    torch.save(checkpoint, approved_path)

    with open(approved_metadata_path, "w") as f:
        json.dump(validation_result, f, indent=2)

    handoff_path = ARTIFACTS_DIR / "approved_checkpoint.json"

    handoff = {
        "approved": True,
        "checkpoint_path": str(approved_path),
        "original_checkpoint": str(checkpoint_path),
        "accuracy": accuracy,
        "min_accuracy": MIN_ACCURACY,
        "status": status,
        "source_pipeline": "Continuous Training Pipeline",
        "next_pipeline": "Model Continuous Delivery Pipeline"
    }

    with open(handoff_path, "w") as f:
        json.dump(handoff, f, indent=2)

    task.upload_artifact(
        name="approved_model",
        artifact_object=str(approved_path)
    )

    task.upload_artifact(
        name="approved_model_metadata",
        artifact_object=str(approved_metadata_path)
    )

    task.upload_artifact(
        name="approved_checkpoint_handoff",
        artifact_object=str(handoff_path)
    )

    print("Validation complete.")
    print(validation_result)
    print(f"Approved model saved to: {approved_path}")
    print(f"Approved checkpoint handoff created: {handoff_path}")


if __name__ == "__main__":
    validate()