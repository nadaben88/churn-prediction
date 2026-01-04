# End-to-End MLOps Churn Prediction Pipeline

**Author**: Nada Bentaouyt
**Project Status**: Active
**Version**: 1.0.0

---

## Overview
This project is an end-to-end MLOps pipeline for predicting customer churn in the telecom industry, achieving 85% recall and 89.5% AUC-ROC using a hyperparameter-tuned XGBoost model. The solution enables the company to proactively identify customers at high risk of churn and support targeted retention actions, helping improve customer experience and reduce revenue loss associated with churn.The pipeline covers data preprocessing, model training, automated testing, inference utilities, and an interactive Gradio demo for real-time predictions and stakeholder validation. It is designed to be reproducible, scalable, and production-ready, with plans to extend it with API deployment, Docker containerization, and monitoring in the future.
## Model Performance
The XGBoost model achieved the following performance metrics on the test set:
   Metric      | Score       |
 |-------------|-------------|
 | **Recall**  | 85.0%       |
 | **Precision**| 55.3%      |
 | **F1 Score**| 67.0%       |
 | **AUC-ROC** | 89.5%       |
 | **Accuracy** | 77.8%       |

### Interpretation:
- **High Recall (85.0%)**: The model effectively identifies **85% of actual churners**, minimizing false negatives and supporting proactive retention strategies.
- **AUC-ROC (89.5%)**: The model has excellent **ranking capability**, distinguishing churners from non-churners with high confidence.
- **Precision-Recall Trade-off**: The balance between precision and recall is typical for imbalanced datasets like churn prediction. Further tuning (e.g., class weights, thresholds) could optimize this trade-off based on business needs.


## Architecture
The pipeline consists of the following components:

1. **Data Preprocessing**: Cleaning, feature engineering, and transformation.
2. **Model Training**: Hyperparameter tuning and training an XGBoost model.
3. **Inference**: Robust prediction utilities for batch and real-time inference.
4. **Testing**: Unit and integration tests for preprocessing and inference.
5. **Demo**: Gradio interface for interactive testing.

**Future Improvements**:
- FastAPI service for serving predictions.
- Docker containerization for deployment.
- Monitoring with Evidently, Prometheus, and Grafana.


---

## Setup Instructions

### Prerequisites
- Python 3.9
- Conda (recommended for environment management)

### Step 1: Clone the Repository
```bash
git clone https://github.com/nadaben88/churn-prediction.git
cd churn-prediction
```
### Step 2: Create the Conda Environment
```bash
conda env create -f environment.yml
conda activate mlops-churn
```
### Step 3: Install requirements
```bash
pip install -r requirements.txt
```
### Step 4: Run the Pipeline
```bash
#run as a package
python -m src.preprocess
python -m src.train
python -m src.eval
python -m src.predict
#optional : run test
pytest tests/test_preprocess.py -v
pytest tests/test_predict.py -v
#Launch the Gradio Demo
python src/gradio_demo.py
```
### Project Structure:
```bash
mlops-churn/
├── config.yaml                # Configuration file
├── environment.yml             # Conda environment
├── data/
│   ├── raw/                    # Raw data
│   ├── processed/             # Processed data
│   ├── train/                  # Training data
│   └── test/                   # Test data
├── artifacts/                  # Saved models and preprocessors
├── src/
│   ├── preprocess.py          # Data preprocessing
│   ├── train.py                # Model training
│   ├── predict.py              # Inference utilities
│   ├── gradio_demo.py          # Gradio demo
│   └── utils.py                # Utility functions
├── tests/
│   ├── test_preprocess.py      # Preprocessing tests
│   └── test_predict.py         # Inference tests
├── logs/                       # Log files
```                 
### Future Improvements
The following features are planned for future development:
API Deployment

FastAPI Service: Deploy the model as a REST API for real-time predictions.
Docker Containerization

Docker Image: Containerize the API service for easy deployment and scalability.
Monitoring

Data Drift Detection: Use Evidently to monitor data drift.
Model Performance Monitoring: Integrate Prometheus and Grafana for real-time monitoring.

🤝 Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Commit your changes (git commit -am 'Add your feature').
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

📬 Contact
For questions or feedback, please contact:
Nada Bentaouyt
[nadabentaouyt@gmail.com]












