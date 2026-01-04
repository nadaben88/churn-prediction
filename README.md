# End-to-End MLOps Churn Prediction Pipeline

**Author**: Nada Bentaouyt
**Project Status**: Active
**Version**: 1.0.0

---

## Overview
This project is an end-to-end MLOps pipeline for predicting customer churn in the telecom industry, achieving 85% recall and 89.5% AUC-ROC using a hyperparameter-tuned XGBoost model. The solution enables the company to proactively identify customers at high risk of churn and support targeted retention actions, helping improve customer experience and reduce revenue loss associated with churn.The pipeline covers data preprocessing, model training, automated testing, inference utilities, and an interactive Gradio demo for real-time predictions and stakeholder validation. It is designed to be reproducible, scalable, and production-ready, supporting reliable deployment and integration into operational decision-making workflows.
---

## Architecture
The pipeline consists of the following components:

1. **Data Preprocessing**: Cleaning, feature engineering, and transformation.
2. **Model Training**: Hyperparameter tuning and training an XGBoost model.
3. **Inference**: Robust prediction utilities for batch and real-time inference.
4. **Testing**: Unit and integration tests for preprocessing and inference.
5. **Demo**: Gradio interface for interactive testing.

---

## Setup Instructions

### Prerequisites
- Python 3.9
- Conda (recommended for environment management)

### Step 1: Clone the Repository
```bash
git clone https://github.com/nadaben88/churn-prediction.git
cd churn-prediction
### Step 2: Create the Conda Environment




