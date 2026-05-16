from clearml import PipelineController

pipe = PipelineController(
    project="Neural-Network-CICD",
    name="Model CD Pipeline",
    version="1.0.0",
    add_pipeline_tags=True,
)

pipe.set_default_execution_queue("default")

pipe.add_step(
    name="validate_checkpoint",
    base_task_project="Neural-Network-CICD",
    base_task_name="Validate Checkpoint",
)

pipe.add_step(
    name="register_model",
    parents=["validate_checkpoint"],
    base_task_project="Neural-Network-CICD",
    base_task_name="Register Approved MNIST CNN Model",
)

pipe.add_step(
    name="deploy_model",
    parents=["register_model"],
    base_task_project="Neural-Network-CICD",
    base_task_name="Deploy Model to Staging",
)

pipe.add_step(
    name="validate_endpoint",
    parents=["deploy_model"],
    base_task_project="Neural-Network-CICD",
    base_task_name="Validate Deployment Endpoint",
)

pipe.start_locally(run_pipeline_steps_locally=True)

print("Model CD Pipeline submitted.")