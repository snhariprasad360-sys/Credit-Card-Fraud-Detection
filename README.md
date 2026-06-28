# Credit Card Fraud Detection using Machine Learning

## Overview

This project aims to detect fraudulent credit card transactions using Machine Learning classification algorithms. By analyzing historical transaction data, the model learns patterns that distinguish legitimate transactions from fraudulent ones. The project compares multiple algorithms and selects the best-performing model based on evaluation metrics.

---

## Objective

- Detect fraudulent credit card transactions.
- Compare the performance of different Machine Learning algorithms.
- Evaluate models using standard classification metrics.
- Select the best model for fraud prediction.

---

## Dataset

The project uses the Credit Card Fraud Detection dataset containing transaction records.
see the datasets in release.

### Features

- Time
- Amount
- V1 – V28 (PCA-transformed features)
- Class (Target Variable)

### Target Variable

- **0** → Legitimate Transaction
- **1** → Fraudulent Transaction

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Machine Learning Algorithms

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

---

## Project Workflow

```
Dataset
    │
    ▼
Data Collection
    │
    ▼
Data Preprocessing
    │
    ▼
Feature Scaling
    │
    ▼
Train-Test Split
    │
    ▼
Model Training
(Logistic Regression,
Decision Tree,
Random Forest)
    │
    ▼
Model Evaluation
    │
    ▼
Best Model Selection
    │
    ▼
Fraud Prediction
```

---

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

## Project Structure

```
Credit-Card-Fraud-Detection/
│
├── dataset/
│   └── creditcard.csv
│
├── src/
│   ├── data_collection.py
│   ├── data_preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── best_model.py
│   ├── predict.py
│   └── main.py
│
├── images/
│   ├── confusion_matrices.png
│   └── model_comparison.png
│
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git
```

Navigate to the project directory:

```bash
cd Credit-Card-Fraud-Detection
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

Execute the main file:

```bash
python main.py
```

---

## Output

The project generates:

- Trained Machine Learning models
- Performance comparison of all algorithms
- Confusion Matrix
- Classification Report
- Fraud prediction for sample transactions
- Model comparison graph

---

## Results

The performance of the following algorithms is compared:

- Logistic Regression
- Decision Tree
- Random Forest

The best-performing model is selected based on the highest **F1 Score**, ensuring a balanced trade-off between Precision and Recall for fraud detection.

---

## Future Enhancements

- Real-time fraud detection
- Deep Learning-based models
- Hyperparameter tuning
- Streamlit web application
- Deployment using Flask or FastAPI
- Integration with banking systems

---

## Author

**Hari Prasad**

Machine Learning Virtual Internship Project

**CodSoft**

---

## License

This project is developed for educational purposes as part of the CodSoft Machine Learning Internship.
