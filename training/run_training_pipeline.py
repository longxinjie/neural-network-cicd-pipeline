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
    name="evaluate_model",
    parents=["train_model"],
    base_task_project="Neural-Network-CICD",
    base_task_name="Evaluate Model",
)

pipe.add_step(
    name="validate_model",
    parents=["evaluate_model"],
    base_task_project="Neural-Network-CICD",
    base_task_name="Validate Model",
)

pipe.add_step(
    name="trigger_model_cd",
    parents=["validate_model"],
    base_task_project="Neural-Network-CICD",
    base_task_name="Trigger Model CD Pipeline",
)

pipe.start_locally(run_pipeline_steps_locally=True)