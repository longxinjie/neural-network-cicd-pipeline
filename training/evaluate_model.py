import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

from clearml import Task

task = Task.init(
    project_name="Neural-Network-CICD",
    task_name="Evaluate Model"
)

logger = task.get_logger()

CHECKPOINT_DIR = Path("artifacts/trained_checkpoints")
OUTPUT_DIR = Path("reports")
FIGURE_DIR = Path("reports/figures")

OUTPUT_DIR.mkdir(exist_ok=True)
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
        raise FileNotFoundError("No trained checkpoints found.")
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = get_latest_checkpoint()
    print(f"Evaluating checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = SimpleCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])

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

    logger.report_scalar(
        title="Evaluation Accuracy",
        series="accuracy",
        value=accuracy,
        iteration=0
    )

    evaluation_result = {
        "checkpoint": str(checkpoint_path),
        "accuracy": round(accuracy, 4),
        "status": "evaluated"
    }

    with open(OUTPUT_DIR / "evaluation_result.json", "w") as f:
        json.dump(evaluation_result, f, indent=2)

    cm = confusion_matrix(all_labels, all_preds)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(range(10))
    )

    confusion_matrix_path = FIGURE_DIR / "confusion_matrix.png"

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax)
    plt.title("Confusion Matrix - Evaluated MNIST CNN")
    plt.tight_layout()
    plt.savefig(confusion_matrix_path)
    plt.close()

    logger.report_image(
        title="Confusion Matrix",
        series="mnist_cnn",
        local_path=str(confusion_matrix_path),
        iteration=0
    )

    report = classification_report(
        all_labels,
        all_preds,
        output_dict=True
    )

    with open(OUTPUT_DIR / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.report_text(
        f"""
# Model Evaluation Summary

- Checkpoint: {checkpoint_path}
- Evaluation Accuracy: {accuracy:.4f}
- Status: evaluated
- Confusion Matrix: {confusion_matrix_path}
"""
    )

    task.upload_artifact("evaluation_result", "reports/evaluation_result.json")
    task.upload_artifact("classification_report", "reports/classification_report.json")
    task.upload_artifact("confusion_matrix", "reports/figures/confusion_matrix.png")

    print("Evaluation complete.")
    print(evaluation_result)


if __name__ == "__main__":
    evaluate()