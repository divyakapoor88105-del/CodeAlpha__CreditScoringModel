# 💳 Credit Scoring System using Machine Learning

## 📌 Overview

The Credit Scoring System is a Machine Learning-based application designed to evaluate an individual's creditworthiness using historical financial indicators. Financial institutions and lending organizations use credit scoring models to assess the risk associated with approving loans, credit cards, and other financial products.

This project simulates a real-world credit risk assessment workflow by generating a financial dataset, training multiple classification algorithms, evaluating model performance, and providing a user-friendly dashboard for real-time creditworthiness prediction.

---

## 🎯 Objective

The primary objective of this project is to predict whether a customer is creditworthy based on their financial profile.

The system analyzes various financial factors and classifies customers into:

* ✅ Creditworthy
* ❌ Not Creditworthy

The project also compares multiple machine learning algorithms and selects the best-performing model based on evaluation metrics.

---

## 🚀 Features

### Machine Learning Models

The project implements and compares:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

### Financial Features Used

The model uses the following customer attributes:

* Age
* Income
* Debt
* Loan Amount
* Credit Utilization
* Payment History
* Employment Years
* Number of Credit Cards
* Existing Loans
* Late Payments
* Savings
* Debt-to-Income Ratio

### Feature Engineering

A custom financial risk feature is created:

Debt-To-Income Ratio

DebtToIncomeRatio = Debt / Income

This ratio is widely used in the financial industry to measure repayment capability.

---

## 📊 Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score

These metrics help assess classification performance and model reliability.

---

## 🖥️ Dashboard Features

The Streamlit dashboard contains three major sections:

### 1. Model Overview Dashboard

Displays:

* Algorithm comparison table
* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Best performing model

### 2. Credit Prediction Dashboard

Allows users to:

* Enter customer financial information
* Predict creditworthiness
* View approval probability
* Analyze customer risk profile

### 3. Analytics Dashboard

Provides:

* Feature importance visualization
* Dataset preview
* Statistical summary
* Data insights

---

## 📂 Project Structure

```text
Credit_Scoring_Model/
│
├── app.py
├── credit_data.csv
├── generate_dataset.py
├── credit_scoring.py
├── README.md
```

---

## ⚙️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-Learn

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib

### Web Dashboard

* Streamlit

---

## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Credit_Scoring_Model.git
cd Credit_Scoring_Model
```

Install required dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib streamlit
```

---

## ▶️ Running the Project

Generate the dataset:

```bash
python generate_dataset.py
```

Run the Streamlit Dashboard:

```bash
python -m streamlit run app.py
```

Open the browser and visit:

```text
http://localhost:8501
```

---

## 📈 Machine Learning Workflow

1. Dataset Generation
2. Data Preprocessing
3. Feature Engineering
4. Train-Test Split
5. Model Training
6. Performance Evaluation
7. Best Model Selection
8. Real-Time Prediction
9. Dashboard Visualization

---

## 💡 Sample Prediction

### Creditworthy Customer

Income: 150000

Debt: 5000

Payment History: 98

Savings: 200000

Late Payments: 0

Result:

```text
Creditworthy
Approval Probability > 90%
```

### High-Risk Customer

Income: 20000

Debt: 100000

Payment History: 20

Savings: 1000

Late Payments: 15

Result:

```text
Not Creditworthy
Low Approval Probability
```

---

## 📚 Future Enhancements

* Real Banking Dataset Integration
* XGBoost and LightGBM Models
* SHAP Explainability
* Loan Approval Recommendation Engine
* Cloud Deployment
* User Authentication
* PDF Credit Reports

---

## 🎓 Learning Outcomes

This project demonstrates:

* Machine Learning Classification
* Credit Risk Analysis
* Financial Data Processing
* Feature Engineering
* Model Evaluation
* Interactive Dashboard Development
* End-to-End ML Project Deployment

---

## 📄 License

This project is developed for educational, internship, and learning purposes.
.

