from clearml import Task, PipelineController

Task.set_reuse_time_window_in_hours(0)

pipe = PipelineController(
    project="Neural-Network-CICD",
    name="Remote Model Continuous Delivery Pipeline",
    version="1.0.0",
    add_pipeline_tags=True,
)

pipe.set_default_execution_queue("default")

pipe.add_step(
    name="train_model",
    base_task_project="Neural-Network-CICD",
    base_task_name="Local CNN Training",
)

pipe.add_step(
    name="validate_checkpoint",
    parents=["train_model"],
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

print("Submitting remote Model CD pipeline to ClearML queue...")
pipe.start(queue="default")
print("Pipeline submitted.")