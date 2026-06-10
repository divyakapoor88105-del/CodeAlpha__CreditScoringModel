import pandas as pd
import numpy as np

np.random.seed(42)

rows = 100000

age = np.random.randint(21, 70, rows)

income = np.random.randint(15000, 250000, rows)

debt = np.random.randint(0, 150000, rows)

loan_amount = np.random.randint(5000, 500000, rows)

credit_utilization = np.random.uniform(0, 1, rows)

payment_history = np.random.randint(0, 101, rows)

employment_years = np.random.randint(0, 40, rows)

num_credit_cards = np.random.randint(0, 15, rows)

existing_loans = np.random.randint(0, 8, rows)

late_payments = np.random.randint(0, 20, rows)

savings = np.random.randint(0, 1000000, rows)

debt_to_income = debt / (income + 1)

score = (
    payment_history * 0.35
    + (1 - credit_utilization) * 100 * 0.20
    + employment_years * 2
    - late_payments * 4
    - debt_to_income * 50
)

creditworthy = (score > 35).astype(int)

df = pd.DataFrame({
    "Age": age,
    "Income": income,
    "Debt": debt,
    "LoanAmount": loan_amount,
    "CreditUtilization": credit_utilization,
    "PaymentHistory": payment_history,
    "EmploymentYears": employment_years,
    "NumCreditCards": num_credit_cards,
    "ExistingLoans": existing_loans,
    "LatePayments": late_payments,
    "Savings": savings,
    "DebtToIncomeRatio": debt_to_income,
    "Creditworthy": creditworthy
})

df.to_csv("credit_data.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print("Shape:", df.shape)