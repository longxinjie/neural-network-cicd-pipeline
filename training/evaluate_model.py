import json
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from clearml import Task


task = Task.init(
    project_name="Neural-Network-CICD",
    task_name="Evaluate Model"
)

logger = task.get_logger()

CHECKPOINT_DIR = Path("artifacts/trained_checkpoints")
OUTPUT_DIR = Path("reports")
FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)


def get_latest_checkpoint():
    checkpoints = list(CHECKPOINT_DIR.glob("*.pt"))

    if not checkpoints:
        raise FileNotFoundError(
            f"No trained checkpoints found in {CHECKPOINT_DIR}. "
            "Run training first."
        )

    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(range(10))
    )

    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix - MNIST CNN")

    plt.tight_layout()

    path = FIGURE_DIR / "confusion_matrix.png"
    fig.savefig(path)
    logger.report_matplotlib_figure(
        title="Evaluation Plots",
        series="Confusion Matrix",
        figure=fig,
        iteration=0
    )

    plt.close(fig)
    return path


def plot_class_accuracy(y_true, y_pred):
    class_accuracy = {}

    for digit in range(10):
        total = sum(1 for label in y_true if label == digit)
        correct = sum(
            1 for label, pred in zip(y_true, y_pred)
            if label == digit and pred == digit
        )

        class_accuracy[digit] = correct / total if total > 0 else 0

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        list(class_accuracy.keys()),
        list(class_accuracy.values())
    )

    ax.set_title("Per-Class Accuracy")
    ax.set_xlabel("Digit Class")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(list(range(10)))
    ax.set_ylim(0, 1)

    plt.tight_layout()

    path = FIGURE_DIR / "per_class_accuracy.png"
    fig.savefig(path)

    logger.report_matplotlib_figure(
        title="Evaluation Plots",
        series="Per-Class Accuracy",
        figure=fig,
        iteration=0
    )

    plt.close(fig)
    return path, class_accuracy


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = get_latest_checkpoint()
    print(f"Evaluating checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = SimpleCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    logger.report_scalar(
        title="Evaluation Metrics",
        series="accuracy",
        value=accuracy,
        iteration=0
    )

    logger.report_scalar(
        title="Evaluation Metrics",
        series="precision_macro",
        value=precision,
        iteration=0
    )

    logger.report_scalar(
        title="Evaluation Metrics",
        series="recall_macro",
        value=recall,
        iteration=0
    )

    logger.report_scalar(
        title="Evaluation Metrics",
        series="f1_macro",
        value=f1,
        iteration=0
    )

    confusion_matrix_path = plot_confusion_matrix(all_labels, all_preds)
    per_class_accuracy_path, class_accuracy = plot_class_accuracy(all_labels, all_preds)

    report = classification_report(
        all_labels,
        all_preds,
        output_dict=True,
        zero_division=0
    )

    evaluation_result = {
        "checkpoint": str(checkpoint_path),
        "accuracy": round(accuracy, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "f1_macro": round(f1, 4),
        "per_class_accuracy": {
            str(k): round(v, 4)
            for k, v in class_accuracy.items()
        },
        "status": "evaluated"
    }

    evaluation_result_path = OUTPUT_DIR / "evaluation_result.json"
    classification_report_path = OUTPUT_DIR / "classification_report.json"

    with open(evaluation_result_path, "w") as f:
        json.dump(evaluation_result, f, indent=2)

    with open(classification_report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.report_text(
        f"""
# Model Evaluation Summary

Checkpoint: {checkpoint_path}

Accuracy: {accuracy:.4f}  
Precision Macro: {precision:.4f}  
Recall Macro: {recall:.4f}  
F1 Macro: {f1:.4f}  

Status: evaluated
"""
    )

    task.upload_artifact("evaluation_result", str(evaluation_result_path))
    task.upload_artifact("classification_report", str(classification_report_path))
    task.upload_artifact("confusion_matrix", str(confusion_matrix_path))
    task.upload_artifact("per_class_accuracy", str(per_class_accuracy_path))

    print("Evaluation complete.")
    print(json.dumps(evaluation_result, indent=2))


if __name__ == "__main__":
    evaluate()