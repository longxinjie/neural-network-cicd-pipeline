import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
from clearml import Task, OutputModel


MODEL_PATH = Path("model_package/approved_model.pt")
METADATA_PATH = Path("model_package/approved_model_metadata.json")

OUTPUT_DIR = Path("artifacts/model_registry")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REGISTRY_SUMMARY_PATH = OUTPUT_DIR / "model_registry_summary.json"
REGISTRY_PLOT_PATH = OUTPUT_DIR / "model_registry_plot.png"


if not MODEL_PATH.exists():
    raise FileNotFoundError("approved_model.pt not found. Run validate_checkpoint.py first.")

if not METADATA_PATH.exists():
    raise FileNotFoundError("approved_model_metadata.json not found. Run validate_checkpoint.py first.")


task = Task.init(
    project_name="Neural-Network-CICD",
    task_name="Register Approved MNIST CNN Model"
)

logger = task.get_logger()


with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

task.connect(metadata)


model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)

registry_summary = {
    "model_name": "approved_mnist_cnn",
    "framework": "PyTorch",
    "registered_at": datetime.now().isoformat(),
    "model_path": str(MODEL_PATH),
    "metadata_path": str(METADATA_PATH),
    "model_size_mb": model_size_mb,
    "registration_status": "registered",
    "metadata": metadata
}

with open(REGISTRY_SUMMARY_PATH, "w") as f:
    json.dump(registry_summary, f, indent=4)


output_model = OutputModel(
    task=task,
    name="approved_mnist_cnn",
    framework="PyTorch"
)

output_model.update_weights(weights_filename=str(MODEL_PATH))


logger.report_scalar(
    title="Model Registry",
    series="model_registered",
    value=1,
    iteration=0
)

logger.report_scalar(
    title="Model Registry",
    series="model_size_mb",
    value=model_size_mb,
    iteration=0
)

for key, value in metadata.items():
    if isinstance(value, (int, float)):
        logger.report_scalar(
            title="Approved Model Metadata",
            series=key,
            value=value,
            iteration=0
        )


plot_metrics = {
    "model_registered": 1,
    "model_size_mb": model_size_mb
}

for key, value in metadata.items():
    if isinstance(value, (int, float)):
        plot_metrics[key] = value

plt.figure(figsize=(8, 5))
plt.bar(plot_metrics.keys(), plot_metrics.values())
plt.title("Model Registry Summary")
plt.ylabel("Value")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(REGISTRY_PLOT_PATH)
plt.close()

logger.report_image(
    title="Model Registry Plot",
    series="Registry Summary",
    local_path=str(REGISTRY_PLOT_PATH),
    iteration=0
)


task.upload_artifact(
    name="approved_model_metadata",
    artifact_object=metadata
)

task.upload_artifact(
    name="model_registry_summary",
    artifact_object=str(REGISTRY_SUMMARY_PATH)
)

task.upload_artifact(
    name="model_registry_plot",
    artifact_object=str(REGISTRY_PLOT_PATH)
)


logger.report_text("Approved model registered in ClearML.")
logger.report_text(json.dumps(registry_summary, indent=4))

print("Approved model registered in ClearML.")
print("Model path:", MODEL_PATH)
print("Metadata:", metadata)

task.close()