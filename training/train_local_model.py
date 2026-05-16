import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


CHECKPOINT_DIR = Path("artifacts/trained_checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_DIR = Path("reports")
METRICS_DIR.mkdir(exist_ok=True)


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

    max_batches = 300  # keeps live demo fast

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

        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}/{max_batches}, Loss: {loss.item():.4f}")

    train_accuracy = correct / total
    avg_loss = running_loss / max_batches

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
        "training_batches": max_batches,
        "train_accuracy": round(train_accuracy, 4),
        "train_loss": round(avg_loss, 4),
        "device": str(device),
        "status": "trained"
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nTraining complete!")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Training loss: {avg_loss:.4f}")


if __name__ == "__main__":
    train()