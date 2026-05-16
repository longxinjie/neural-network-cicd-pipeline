import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from clearml import Task

task = Task.init(
    project_name="Neural-Network-CICD",
    task_name="Validate Checkpoint"
)

CHECKPOINT_DIR = Path("artifacts/trained_checkpoints")
OUTPUT_DIR = Path("reports")
FIGURE_DIR = Path("reports/figures")
MODEL_PACKAGE_DIR = Path("model_package")

OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PACKAGE_DIR.mkdir(exist_ok=True)

MIN_ACCURACY = 0.80


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
        raise FileNotFoundError("No trained checkpoints found.")

    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def validate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = get_latest_checkpoint()
    print(f"Validating checkpoint: {checkpoint_path}")

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

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total

    status = "PASSED" if accuracy >= MIN_ACCURACY else "FAILED"

    validation_result = {
        "checkpoint": str(checkpoint_path),
        "accuracy": round(accuracy, 4),
        "min_accuracy": MIN_ACCURACY,
        "status": status
    }

    with open(OUTPUT_DIR / "validation_result.json", "w") as f:
        json.dump(validation_result, f, indent=2)

    # Confusion matrix visual
    cm = confusion_matrix(all_labels, all_preds)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(range(10))
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax)
    plt.title("Confusion Matrix - Validated MNIST CNN")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix.png")
    plt.close()

    # Classification report
    report = classification_report(
        all_labels,
        all_preds,
        output_dict=True
    )

    with open(OUTPUT_DIR / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Promotion decision visual
    plt.figure(figsize=(6, 4))
    plt.bar(["Accuracy", "Threshold"], [accuracy, MIN_ACCURACY])
    plt.ylim(0, 1)
    plt.title(f"Model Validation Decision: {status}")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "validation_decision.png")
    plt.close()

    if status == "FAILED":
        raise ValueError(
            f"Model rejected. Accuracy {accuracy:.4f} < threshold {MIN_ACCURACY}"
        )

    # Copy approved checkpoint to model package
    approved_path = MODEL_PACKAGE_DIR / "approved_model.pt"
    torch.save(checkpoint, approved_path)

    with open(MODEL_PACKAGE_DIR / "approved_model_metadata.json", "w") as f:
        json.dump(validation_result, f, indent=2)

    print("Validation complete.")
    print(validation_result)
    print(f"Approved model saved to: {approved_path}")

    task.upload_artifact("validation_result", "reports/validation_result.json")
    task.upload_artifact("classification_report", "reports/classification_report.json")
    task.upload_artifact("confusion_matrix", "reports/figures/confusion_matrix.png")
    task.upload_artifact("validation_decision", "reports/figures/validation_decision.png")

if __name__ == "__main__":
    validate()