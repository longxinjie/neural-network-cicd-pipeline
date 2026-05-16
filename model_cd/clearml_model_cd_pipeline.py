from clearml import PipelineController

pipe = PipelineController(
    project="Neural-Network-CICD",
    name="Remote Model Continuous Delivery Pipeline",
    version="1.0.1",
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

print("Submitting remote Model CD pipeline to ClearML queue...")
pipe.start_locally(run_pipeline_steps_locally=True)
print("Pipeline submitted.")