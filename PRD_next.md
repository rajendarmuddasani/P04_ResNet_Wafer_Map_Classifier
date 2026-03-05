# PRD_next: MLOps Enhancements for ResNet Wafer Map Classifier

## 1. Project Overview

This project, the ResNet Wafer Map Classifier, is a highly successful implementation of a U-Net architecture with a ResNet-50 backbone for wafer map defect pattern classification. It leverages active learning and transfer learning to achieve an impressive 96% accuracy across 25,000 wafer maps and 9 distinct defect patterns. The model is reportedly deployed in production and has demonstrated a significant return on investment (>$2M). While the core model is robust and effective, this document outlines a roadmap for enhancing its MLOps capabilities to improve scalability, maintainability, and readiness for modern, automated ML workflows. These enhancements will also showcase a strong understanding of production-grade ML engineering, which is critical for senior ML engineering roles.

## 2. Current State Assessment

| Feature | Status | Notes |
| :--- | :--- | :--- |
| **Core Model** | **Exists** | PyTorch implementation of U-Net/ResNet-50 with active learning. High accuracy. |
| **Data Processing** | **Exists** | Includes an STDF parser for handling wafer map data. |
| **README.md** | **Exists** | Basic documentation is present. |
| **requirements.txt** | **Exists** | Dependencies are listed. |
| **Model Serving** | **Missing** | No REST API for inference (e.g., FastAPI/Flask). |
| **Containerization** | **Missing** | No Dockerfile or docker-compose.yaml for reproducible environments. |
| **Experiment Tracking** | **Missing** | No integration with MLflow or similar tools for tracking experiments. |
| **Model Monitoring** | **Missing** | No continuous monitoring for data drift or model degradation (e.g., Evidently AI). |
| **CI/CD Pipelines** | **Missing** | No GitHub Actions for automated testing, training, or deployment. |
| **Cloud Deployment** | **Missing** | No specific configurations for AWS SageMaker, Azure ML, or other cloud platforms. |
| **Orchestration** | **Missing** | No Airflow DAGs for pipeline orchestration. |
| **API Testing** | **Missing** | No tests for the non-existent API. |

## 3. Gap Analysis and Recommended Improvements

Here is a prioritized list of gaps and detailed recommendations for addressing them.

### 3.1. Model Serving & Containerization (High Priority)

*   **Gap:** The model is not exposed as an API, and the project is not containerized, making deployment and integration difficult.
*   **Recommendation:**
    1.  **Create a FastAPI Application:** Develop a simple REST API with an endpoint that accepts a wafer map (e.g., as a JSON array or image file) and returns the predicted defect pattern.
    2.  **Develop a Dockerfile:** Create a `Dockerfile` to containerize the FastAPI application, including all necessary dependencies from `requirements.txt`. This ensures a consistent and reproducible runtime environment.
    3.  **Use docker-compose:** Add a `docker-compose.yaml` to manage the application container, making it easy to run locally.
*   **Estimated Effort:** 2-3 days

### 3.2. Experiment Tracking with MLflow (High Priority)

*   **Gap:** Lack of experiment tracking makes it difficult to compare model versions, reproduce results, and manage the model lifecycle.
*   **Recommendation:**
    1.  **Integrate MLflow:** Add MLflow logging to the training script to log parameters, metrics (accuracy, loss), and the trained model as an artifact.
    2.  **Use the MLflow Model Registry:** After training, register the best model in the MLflow Model Registry to version and manage it.
*   **Estimated Effort:** 1-2 days

### 3.3. CI/CD with GitHub Actions (Medium Priority)

*   **Gap:** The entire workflow is manual. There is no automation for testing, training, or deployment.
*   **Recommendation:**
    1.  **Create a Training Workflow:** Set up a GitHub Actions workflow (`.github/workflows/training.yml`) that triggers on pushes to the `main` branch. This workflow should run the training script and, upon successful completion, register the model in MLflow.
    2.  **Create a Deployment Workflow:** Create a second workflow (`.github/workflows/deploy.yml`) that triggers after the training workflow. This workflow should build the Docker image, push it to a container registry (like Docker Hub or AWS ECR), and deploy it to a target environment (e.g., AWS SageMaker).
*   **Estimated Effort:** 2-3 days

### 3.4. Cloud Deployment to AWS SageMaker (Medium Priority)

*   **Gap:** The project lacks a clear path to cloud deployment, limiting its scalability and production readiness.
*   **Recommendation:**
    1.  **Configure SageMaker Endpoint:** Adapt the FastAPI application to be compatible with AWS SageMaker. This involves creating a script that SageMaker can use to host the model.
    2.  **Create Deployment Scripts:** Write scripts (e.g., using the AWS CLI or Boto3) to create a SageMaker model, endpoint configuration, and endpoint. This can be integrated into the deployment GitHub Action.
*   **Estimated Effort:** 3-4 days

### 3.5. Model Monitoring with Evidently AI (Low Priority)

*   **Gap:** No system is in place to monitor the model in production for data drift or performance degradation.
*   **Recommendation:**
    1.  **Generate Reference Data:** Create a reference dataset from the training data.
    2.  **Integrate Evidently AI:** In the FastAPI application, add a hook to log incoming prediction data. Periodically, run a script that uses Evidently AI to compare the production data with the reference data and generate a monitoring dashboard.
*   **Estimated Effort:** 2-3 days

## 4. Interview Topics Demonstrated

After implementing these improvements, this project will be an excellent talking point for the following interview topics:

*   **MLOps & Production ML:** Demonstrates a full end-to-end MLOps workflow, from experiment tracking to automated deployment and monitoring.
*   **Model Serving & Containerization:** Shows proficiency with FastAPI and Docker for creating scalable, production-ready services.
*   **Cloud-Native ML:** Highlights experience with deploying ML models on a major cloud platform like AWS SageMaker.
*   **CI/CD for Machine Learning:** Showcases the ability to build automated pipelines for ML systems using GitHub Actions.

## 5. Cloud Deployment Plan (AWS)

1.  **Container Registry:** Use **Amazon Elastic Container Registry (ECR)** to store the Docker images.
2.  **Model Hosting:** Deploy the containerized model to an **AWS SageMaker Endpoint**. SageMaker is ideal for this computer vision model due to its scalability and built-in features for hosting and monitoring.
3.  **Automation:** The entire deployment process will be automated via the **GitHub Actions** workflow, which will be triggered after successful training and model registration in MLflow.

## 6. Quick Win (1-2 Days)

The single most impactful improvement that can be achieved in 1-2 days is to **containerize the model with a FastAPI interface**. This immediately makes the model deployable as a standalone service and provides a foundation for all other MLOps improvements. It\'s a high-impact, low-effort first step that dramatically increases the project\'s practical value.
