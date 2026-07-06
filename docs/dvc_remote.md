# DVC Remote Storage

For this phase, DVC is used to version the raw dataset and reproduce the machine learning pipeline.

The project includes:

- Raw dataset tracked using DVC
- `dvc.yaml` pipeline with prepare, train, and evaluate stages
- `dvc.lock` for reproducibility
- Local DVC cache for data versioning

A cloud remote can be configured later using:

```bash
dvc remote add -d storage <remote-url>
dvc push