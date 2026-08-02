# Model Card: Credit Card Fraud Detection

## Model Overview

| Item | Description |
|---|---|
| Model name | Credit Card Fraud Detection Model |
| Model type | Logistic Regression |
| Task | Binary classification |
| Framework | Scikit-learn |
| API framework | FastAPI |
| Deployment platform | Render |
| Version | 1.0 |
| Project | MAI201 MLOps Project |
| Team | Group 09 |
| Authors | Soodeh Vanaki, Ryan Caezar Soria, Anurag Singh |
| Last updated | August 2026 |

## Model Description

This model predicts whether a credit card transaction is legitimate or fraudulent. It is part of an end-to-end MLOps pipeline that includes data versioning, preprocessing, experiment tracking, automated testing, cloud deployment, drift monitoring, retraining, and conditional model promotion.

The model produces:

- A binary prediction:
  - `0`: legitimate transaction
  - `1`: fraudulent transaction
- A probability representing the estimated likelihood of fraud

## Intended Use

### Intended Users

The model is intended for:

- Academic demonstration of an end-to-end MLOps workflow
- Students and instructors evaluating machine learning deployment practices
- Developers studying reproducible model training, monitoring, and retraining
- Fraud-analysis teams using the project as a prototype or learning reference

### Intended Applications

The model may be used to:

- Demonstrate credit card fraud classification
- Test API-based machine learning inference
- Demonstrate data drift monitoring
- Evaluate automated retraining and model promotion workflows
- Support educational analysis of imbalanced classification

### Out-of-Scope Uses

The model should not be used as the sole decision-maker for:

- Blocking real financial transactions
- Accusing individuals of fraud
- Making legal or financial enforcement decisions
- Production banking systems without additional validation, security, governance, and human review

## Dataset

The project uses the Credit Card Fraud Detection dataset.

### Dataset Characteristics

| Property | Value |
|---|---:|
| Total transactions | 284,807 |
| Input features | 30 |
| Target variable | `Class` |
| Fraudulent transactions | 492 |
| Fraud proportion | Approximately 0.17% |

### Features

- `Time`: seconds elapsed since the first recorded transaction
- `Amount`: transaction amount
- `V1`–`V28`: PCA-transformed numerical features
- `Class`: target variable

### Data Quality and Preprocessing

The preprocessing pipeline:

- Checks the raw dataset
- Removes 1,081 duplicate rows
- Preserves the highly imbalanced target distribution through stratified splitting
- Creates training, validation, and test datasets
- Scales `Time` and `Amount` using `RobustScaler`
- Fits the scaler only on the training set to reduce data leakage
- Applies the fitted scaler to validation, test, and inference data
- Saves the fitted scaler for consistent model serving

After duplicate removal, the dataset contains 283,726 transactions.

## Data Splitting

The cleaned dataset is divided into:

- Training set
- Validation set
- Test set

Stratified splitting is used to preserve the small fraud proportion across all subsets.

The validation set is used for experiment comparison and candidate-model promotion decisions. The test set is used for final model evaluation.

## Model Architecture

The production model is a Scikit-learn Logistic Regression classifier.

### Key Configuration

- Class weighting: balanced
- Random state: 42
- Maximum iterations: configured through `params.yaml`
- Regularization strength: selected through tracked MLflow experiments
- Solver: `lbfgs`

Multiple Logistic Regression configurations were evaluated through MLflow before selecting the production configuration.

## Evaluation Metrics

The following results were obtained on the held-out test dataset:

| Metric | Value |
|---|---:|
| Accuracy | 0.9724 |
| Precision | 0.0507 |
| Recall | 0.8737 |
| F1-score | 0.0958 |
| ROC-AUC | 0.9644 |
| PR-AUC | 0.6884 |
| True negatives | 55,096 |
| False positives | 1,555 |
| False negatives | 12 |
| True positives | 83 |

## Metric Interpretation

The dataset is extremely imbalanced, so accuracy alone is not an appropriate measure of model quality.

The model achieves high recall, identifying approximately 87% of fraudulent transactions in the test set. However, precision is low because many legitimate transactions are classified as suspicious.

PR-AUC is used as the primary model-promotion metric because it is more informative than accuracy for highly imbalanced binary classification.

## Decision Threshold

The classifier currently uses the default probability threshold of `0.5` to convert fraud probabilities into binary predictions.

In a real financial system, this threshold should be selected based on:

- Cost of false positives
- Cost of false negatives
- Fraud investigation capacity
- Customer experience requirements
- Business risk tolerance

## API Interface

The model is served through a FastAPI REST API.

### Public Service

Base URL:

`https://credit-card-fraud-mlops-jy3a.onrender.com`

Interactive documentation:

`https://credit-card-fraud-mlops-jy3a.onrender.com/docs`

### Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Model and scaler health check |
| GET | `/version` | API version |
| GET | `/model-info` | Model metadata |
| POST | `/predict` | Fraud prediction and probability |

The prediction endpoint expects all 30 input features. Pydantic validation rejects missing, unexpected, or invalid input values.

## Deployment

The API is:

- Containerized with Docker
- Deployed to Render
- Bound to the cloud platform’s assigned port
- Automatically redeployed when changes are committed to the monitored `main` branch
- Tested through Swagger UI and automated API tests

Small deployment copies of the trained model and scaler are packaged with the Docker service to support cloud startup without requiring access to the private DVC remote.

## Monitoring

Evidently AI is used to compare reference data against current data.

### Baseline Scenario

| Item | Result |
|---|---:|
| Drifted features | 0 / 30 |
| Drift share | 0.00% |
| Retraining recommended | No |

### Simulated Drift Scenario

| Item | Result |
|---|---:|
| Drifted features | 20 / 30 |
| Drift share | 64.52% |
| Retraining recommended | Yes |

The simulated scenario intentionally changes the distributions of `Time`, `Amount`, and `V1`–`V18`. It is used only to demonstrate the monitoring and retraining workflow and does not represent observed production behaviour.

## Retraining Policy

When Evidently detects drift above the configured threshold:

1. A machine-readable retraining decision is generated.
2. Historical training data is combined with newly available labeled current data.
3. A candidate model is trained using the configured production parameters.
4. The current and candidate models are evaluated on the same validation dataset.
5. The candidate is promoted only when:
   - Its PR-AUC is equal to or greater than the current model’s PR-AUC.
   - Its recall remains within the configured tolerance.
6. If the candidate fails either condition, it is rejected and the production model remains unchanged.
7. The retraining run, metrics, decision, and artifacts are logged in MLflow.

During the demonstrated retraining run, the candidate model was rejected, confirming that the promotion safeguard prevented a weaker model from replacing the production model.

## Model Versioning and Promotion

The workflow distinguishes between:

- `fraud_model.pkl`: current production model
- `candidate_model.pkl`: newly retrained candidate
- Archived production model: rollback copy created before a successful promotion

The production model is never overwritten before candidate evaluation.

## Testing and Validation

The project includes automated tests for:

- API endpoints
- Input validation
- Data preprocessing and validation
- Monitoring utilities
- Drift-summary extraction
- Model comparison
- Retraining utilities
- Candidate promotion and rollback behavior

The final local test suite contains 39 passing tests.

The CI pipeline also performs:

- Ruff linting
- Ruff formatting checks
- Data-validation tests
- Full Pytest test suite
- Docker image build

## Limitations

- The model was trained on one public dataset.
- The PCA-transformed features are anonymized, limiting interpretability.
- The dataset contains transactions from a limited historical period.
- Fraud patterns may change across institutions, countries, customer groups, and time.
- Low precision may produce many false-positive alerts.
- The simulated drift scenario does not represent actual production traffic.
- The model does not currently provide feature-level explanations.
- The default classification threshold has not been optimized for a specific business cost function.
- Cloud deployment uses a free service tier and may experience cold-start delays.
- The system is an academic prototype and has not undergone financial-industry security or compliance review.

## Ethical and Risk Considerations

### False Positives

Legitimate transactions may be flagged as fraudulent. In a real system, this could inconvenience customers or delay valid purchases.

### False Negatives

Fraudulent transactions may be missed, creating financial loss and security risks.

### Human Oversight

Predictions should support human investigation rather than independently determine customer consequences.

### Fairness

The anonymized dataset does not provide demographic information, so fairness across protected groups cannot be evaluated.

### Privacy and Security

The public dataset contains anonymized features. A production system would require secure storage, access controls, encryption, audit logging, and compliance with relevant financial and privacy regulations.

## Recommended Human Oversight

In production, fraud predictions should be used as risk signals. High-risk transactions should be reviewed using additional evidence, business rules, and qualified human judgment.

## Reproducibility

The project supports reproducibility through:

- Git and GitHub version control
- DVC data and artifact versioning
- A three-stage DVC pipeline
- MLflow experiment tracking
- Versioned parameters in `params.yaml`
- Docker containerization
- Automated tests and CI
- Documented API and deployment configuration

## Known Risks

| Risk | Mitigation |
|---|---|
| High false-positive rate | Tune threshold and use human review |
| Data drift | Evidently monitoring |
| Poor retrained model | Conditional promotion checks |
| Deployment failure | Docker health checks and Render logs |
| Reproducibility failure | DVC, MLflow, configuration files, and tests |
| Model degradation | Validation-based candidate comparison |
| Incorrect API input | Pydantic schema validation |

## Future Improvements

- Evaluate Random Forest and XGBoost models
- Tune the probability threshold based on business costs
- Add explainability using SHAP or similar methods
- Add scheduled monitoring with secure DVC access
- Use the MLflow Model Registry for production version management
- Capture real production inference data for monitoring
- Add authentication and rate limiting to the API
- Add alerting for drift and failed deployments
- Evaluate fairness if appropriate demographic data becomes available

## Version History

| Version | Description |
|---|---|
| 1.0 | Logistic Regression baseline, DVC pipeline, MLflow tracking, Docker, and CI |
| 2.0 | FastAPI serving, Render deployment, Evidently monitoring, drift-triggered retraining, and conditional promotion |

## Contact and Repository

Repository:

`https://github.com/svanaki/credit-card-fraud-mlops`

Public API documentation:

`https://credit-card-fraud-mlops-jy3a.onrender.com/docs`