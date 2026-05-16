# Neural Network CI/CD Pipeline

An end-to-end MLOps workflow for training, validating, registering, deploying, serving and monitoring a neural-network model.

## Project Objective

This project demonstrates how CI/CD concepts can be applied to machine learning systems, where the deployed artifact is not only software code but also a trained neural-network checkpoint.

## Architecture

| Stage | Tool |
|---|---|
| Local neural-network training | PyTorch |
| Pipeline CI | GitHub Actions |
| Visual CI reporting | CML |
| Model registry and tracking | ClearML |
| Local staging deployment | Flask |
| Monitoring visuals | Matplotlib |

## Workflow

```text
Local Training
      ↓
Checkpoint Validation
      ↓
CML Visual Report
      ↓
ClearML Model Registration
      ↓
Local Staging Deployment
      ↓
Flask Inference Endpoint
      ↓
Monitoring & Feedback Loop