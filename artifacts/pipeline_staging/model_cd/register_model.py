import json
from pathlib import Path
from clearml import Task, OutputModel

MODEL_PATH = Path("model_package/approved_model.pt")
METADATA_PATH = Path("model_package/approved_model_metadata.json")

if not MODEL_PATH.exists():
    raise FileNotFoundError("approved_model.pt not found. Run validate_checkpoint.py first.")

if not METADATA_PATH.exists():
    raise FileNotFoundError("approved_model_metadata.json not found. Run validate_checkpoint.py first.")

task = Task.init(
    project_name="Neural-Network-CICD",
    task_name="Register Approved MNIST CNN Model"
)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

task.connect(metadata)

output_model = OutputModel(
    task=task,
    name="approved_mnist_cnn",
    framework="PyTorch"
)

output_model.update_weights(weights_filename=str(MODEL_PATH))

task.upload_artifact(
    name="approved_model_metadata",
    artifact_object=metadata
)

print("Approved model registered in ClearML.")
print("Model path:", MODEL_PATH)
print("Metadata:", metadata)