from clearml import PipelineController

pipe = PipelineController(
    project="Neural-Network-CI/CD",
    name="Continuous Training Pipeline",
    version="1.0.0",
    add_pipeline_tags=True,
)

pipe.set_default_execution_queue("default")

pipe.add_step(
    name="simulate_new_data",
    base_task_project="Neural-Network-CICD",
    base_task_name="Simulate New Data"
)

pipe.add_step(
    name="check_retrain_trigger",
    parents=["simulate_new_data"],
    base_task_project="Neural-Network-CICD",
    base_task_name="Check Retrain Trigger"
)

pipe.add_step(
    name="train_cnn",
    parents=["check_retrain_trigger"],
    base_task_project="Neural-Network-CICD",
    base_task_name="Local CNN Training"
)

pipe.add_step(
    name="validate_checkpoint",
    parents=["train_cnn"],
    base_task_project="Neural-Network-CICD",
    base_task_name="Validate Checkpoint"
)

pipe.start_locally(run_pipeline_steps_locally=True)