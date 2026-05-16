import subprocess
import sys

steps = [
    ("Validate checkpoint", "python model_cd/validate_checkpoint.py"),
    ("Register model in ClearML", "python model_cd/register_model.py"),
    ("Deploy model to staging", "python model_cd/deploy_model.py"),
]

for name, command in steps:
    print(f"\n=== {name} ===")
    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print(f"Step failed: {name}")
        sys.exit(result.returncode)

print("\nModel CD pipeline completed successfully.")