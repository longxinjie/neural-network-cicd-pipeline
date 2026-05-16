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
FIGURE_DIR = Path("reports/figures")
MODEL_PACKAGE_DIR = Path("model_package")
ARTIFACTS_DIR = Path("artifacts")

OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PACKAGE_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

MIN_ACCURACY = 0.80


def validate():
    evaluation_path = OUTPUT_DIR / "evaluation_result.json"

    if not evaluation_path.exists():
        raise FileNotFoundError("No evaluation_result.json found. Run evaluation first.")

    with open(evaluation_path, "r") as f:
        evaluation_result = json.load(f)

    checkpoint_path = Path(evaluation_result["checkpoint"])
    accuracy = evaluation_result["accuracy"]

    status = "PASSED" if accuracy >= MIN_ACCURACY else "FAILED"

    validation_result = {
        "checkpoint": str(checkpoint_path),
        "accuracy": accuracy,
        "min_accuracy": MIN_ACCURACY,
        "status": status
    }

    with open(OUTPUT_DIR / "validation_result.json", "w") as f:
        json.dump(validation_result, f, indent=2)

    validation_decision_path = FIGURE_DIR / "validation_decision.png"

    plt.figure(figsize=(6, 4))
    plt.bar(["Accuracy", "Threshold"], [accuracy, MIN_ACCURACY])
    plt.ylim(0, 1)
    plt.title(f"Model Validation Decision: {status}")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(validation_decision_path)
    plt.close()

    logger.report_scalar(
        title="Validation Accuracy",
        series="accuracy",
        value=accuracy,
        iteration=0
    )

    logger.report_scalar(
        title="Validation Threshold",
        series="min_accuracy",
        value=MIN_ACCURACY,
        iteration=0
    )

    logger.report_image(
        title="Validation Decision",
        series="accuracy_vs_threshold",
        local_path=str(validation_decision_path),
        iteration=0
    )

    if status == "FAILED":
        logger.report_text(
            f"""
# Model Validation Summary

- Status: FAILED
- Accuracy: {accuracy:.4f}
- Required Threshold: {MIN_ACCURACY}
- Decision: Model rejected
"""
        )

        task.upload_artifact("validation_result", "reports/validation_result.json")
        task.upload_artifact("validation_decision", str(validation_decision_path))

        raise ValueError(
            f"Model rejected. Accuracy {accuracy:.4f} < threshold {MIN_ACCURACY}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    approved_path = MODEL_PACKAGE_DIR / "approved_model.pt"
    torch.save(checkpoint, approved_path)

    with open(MODEL_PACKAGE_DIR / "approved_model_metadata.json", "w") as f:
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

    logger.report_text(
        f"""
# Model Validation Summary

- Status: PASSED
- Accuracy: {accuracy:.4f}
- Required Threshold: {MIN_ACCURACY}
- Approved Model: {approved_path}
- Handoff Artifact: {handoff_path}
"""
    )

    task.upload_artifact("validation_result", "reports/validation_result.json")
    task.upload_artifact("validation_decision", str(validation_decision_path))
    task.upload_artifact("approved_model", str(approved_path))
    task.upload_artifact("approved_checkpoint_handoff", str(handoff_path))

    print("Validation complete.")
    print(validation_result)
    print(f"Approved model saved to: {approved_path}")
    print(f"Approved checkpoint handoff created: {handoff_path}")


if __name__ == "__main__":
    validate()