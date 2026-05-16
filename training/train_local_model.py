import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from clearml import Task


Task.force_requirements_env_freeze(
    requirements_file="requirements.txt"
)

task = Task.init(
    project_name="Neural-Network-CICD",
    task_name="Local CNN Training"
)

logger = task.get_logger()

CHECKPOINT_DIR = Path("artifacts/trained_checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = Path("reports")
FIGURE_DIR = REPORTS_DIR / "figures"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
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


def save_training_curve(iterations, batch_losses, avg_losses, running_accuracies):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(iterations, batch_losses, label="Batch Loss")
    ax.plot(iterations, avg_losses, label="Average Loss")
    ax.set_title("Training Loss Curve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    loss_curve_path = FIGURE_DIR / "training_loss_curve.png"
    fig.tight_layout()
    fig.savefig(loss_curve_path)

    logger.report_matplotlib_figure(
        title="Training Plots",
        series="Training Loss Curve",
        figure=fig,
        iteration=0
    )

    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(iterations, running_accuracies, label="Running Accuracy")
    ax.set_title("Training Accuracy Curve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)

    accuracy_curve_path = FIGURE_DIR / "training_accuracy_curve.png"
    fig.tight_layout()
    fig.savefig(accuracy_curve_path)

    logger.report_matplotlib_figure(
        title="Training Plots",
        series="Training Accuracy Curve",
        figure=fig,
        iteration=0
    )

    plt.close(fig)

    return loss_curve_path, accuracy_curve_path


def save_sample_predictions(model, dataset, device):
    model.eval()

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))

    with torch.no_grad():
        for i, ax in enumerate(axes.flat):
            image, label = dataset[i]
            image_input = image.unsqueeze(0).to(device)

            output = model(image_input)
            predicted = torch.argmax(output, dim=1).item()

            ax.imshow(image.squeeze(), cmap="gray")
            ax.set_title(f"Pred: {predicted} | True: {label}")
            ax.axis("off")

    fig.suptitle("Sample Training Predictions")
    fig.tight_layout()

    sample_predictions_path = FIGURE_DIR / "sample_training_predictions.png"
    fig.savefig(sample_predictions_path)

    logger.report_matplotlib_figure(
        title="Training Plots",
        series="Sample Predictions",
        figure=fig,
        iteration=0
    )

    plt.close(fig)

    model.train()

    return sample_predictions_path


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    iterations = []
    batch_losses = []
    avg_losses = []
    running_accuracies = []

    max_batches = 300

    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        current_accuracy = correct / total
        current_avg_loss = running_loss / (batch_idx + 1)

        iterations.append(batch_idx)
        batch_losses.append(loss.item())
        avg_losses.append(current_avg_loss)
        running_accuracies.append(current_accuracy)

        logger.report_scalar(
            title="Training Loss",
            series="batch_loss",
            value=loss.item(),
            iteration=batch_idx
        )

        logger.report_scalar(
            title="Training Loss",
            series="average_loss",
            value=current_avg_loss,
            iteration=batch_idx
        )

        logger.report_scalar(
            title="Training Accuracy",
            series="running_accuracy",
            value=current_accuracy,
            iteration=batch_idx
        )

        if batch_idx % 50 == 0:
            print(
                f"Batch {batch_idx}/{max_batches}, "
                f"Loss: {loss.item():.4f}, "
                f"Running Accuracy: {current_accuracy:.4f}"
            )

    actual_batches = len(iterations)

    train_accuracy = correct / total
    avg_loss = running_loss / actual_batches

    loss_curve_path, accuracy_curve_path = save_training_curve(
        iterations,
        batch_losses,
        avg_losses,
        running_accuracies
    )

    sample_predictions_path = save_sample_predictions(
        model=model,
        dataset=train_dataset,
        device=device
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint_path = CHECKPOINT_DIR / f"mnist_cnn_checkpoint_{timestamp}.pt"
    metadata_path = CHECKPOINT_DIR / f"mnist_cnn_checkpoint_{timestamp}.json"

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_architecture": "SimpleCNN",
        "created_at": timestamp,
        "input_shape": [1, 28, 28],
        "num_classes": 10
    }, checkpoint_path)

    metadata = {
        "checkpoint": str(checkpoint_path),
        "created_at": timestamp,
        "model_architecture": "SimpleCNN",
        "training_batches": actual_batches,
        "train_accuracy": round(train_accuracy, 4),
        "train_loss": round(avg_loss, 4),
        "device": str(device),
        "plots": {
            "training_loss_curve": str(loss_curve_path),
            "training_accuracy_curve": str(accuracy_curve_path),
            "sample_predictions": str(sample_predictions_path)
        },
        "status": "trained"
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # =========================
    # Extra files for retrain_trigger.py
    # =========================
    training_metrics = {
        "train_accuracy": float(train_accuracy),
        "train_loss": float(avg_loss),
        "training_batches": int(actual_batches)
    }

    training_metrics_path = ARTIFACT_DIR / "training_metrics.json"

    with open(training_metrics_path, "w") as f:
        json.dump(training_metrics, f, indent=4)

    history_df = pd.DataFrame({
        "epoch": iterations,
        "batch_loss": batch_losses,
        "average_loss": avg_losses,
        "running_accuracy": running_accuracies
    })

    training_history_path = ARTIFACT_DIR / "training_history.csv"
    history_df.to_csv(training_history_path, index=False)

    task.upload_artifact(
        name="training_metrics_json",
        artifact_object=str(training_metrics_path)
    )

    task.upload_artifact(
        name="training_history_csv",
        artifact_object=str(training_history_path)
    )

    logger.report_text(
        f"""
# CNN Training Summary

- Training Accuracy: {train_accuracy:.4f}
- Training Loss: {avg_loss:.4f}
- Checkpoint: {checkpoint_path}
- Metadata: {metadata_path}
- Metrics JSON: {training_metrics_path}
- Training History CSV: {training_history_path}
- Loss Curve: {loss_curve_path}
- Accuracy Curve: {accuracy_curve_path}
- Sample Predictions: {sample_predictions_path}
- Device: {device}
- Status: trained
"""
    )

    task.upload_artifact(
        name="trained_checkpoint",
        artifact_object=str(checkpoint_path)
    )

    task.upload_artifact(
        name="training_metadata",
        artifact_object=str(metadata_path)
    )

    task.upload_artifact(
        name="training_summary",
        artifact_object=metadata
    )

    task.upload_artifact(
        name="training_loss_curve",
        artifact_object=str(loss_curve_path)
    )

    task.upload_artifact(
        name="training_accuracy_curve",
        artifact_object=str(accuracy_curve_path)
    )

    task.upload_artifact(
        name="sample_training_predictions",
        artifact_object=str(sample_predictions_path)
    )

    print("\nTraining complete!")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Metrics saved to: {training_metrics_path}")
    print(f"History saved to: {training_history_path}")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Training loss: {avg_loss:.4f}")

    task.close()


if __name__ == "__main__":
    train()