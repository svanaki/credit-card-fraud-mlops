# System Architecture

## Pipeline Overview

The system follows a batch machine learning pipeline:

Raw Data → Prepare → Train → Evaluate → Track Experiments

## Technology Choices

- Python: main programming language
- Pandas: data processing
- Scikit-learn: baseline machine learning model
- DVC: data versioning and pipeline reproducibility
- MLflow: experiment tracking and artifact logging
- Docker: reproducible runtime environment
- GitHub Actions: automated testing and Docker build validation

## Deployment Strategy

The current Phase 1 implementation uses a batch pipeline strategy.

In Phase 2, the trained model can be deployed as an online inference service using FastAPI and Docker.