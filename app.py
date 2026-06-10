import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Credit Scoring Dashboard",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Scoring Dashboard")
st.markdown("Machine Learning Based Creditworthiness Prediction")

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():
    return pd.read_csv("credit_data.csv")

df = load_data()

# ==================================================
# VALIDATION
# ==================================================

if "Creditworthy" not in df.columns:
    st.error(
        "Column 'Creditworthy' not found in credit_data.csv"
    )
    st.stop()

# ==================================================
# PREPARE DATA
# ==================================================

X = df.drop("Creditworthy", axis=1)
y = df["Creditworthy"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==================================================
# TRAIN MODELS
# ==================================================

@st.cache_resource
def train_models():

    models = {
        "Logistic Regression":
            LogisticRegression(max_iter=5000),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=8,
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
    }

    results = []

    best_model = None
    best_auc = 0

    for name, model in models.items():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results.append([
            name,
            acc,
            prec,
            rec,
            f1,
            auc
        ])

        if auc > best_auc:
            best_auc = auc
            best_model = model

    results_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC AUC"
        ]
    )

    return best_model, results_df

best_model, results_df = train_models()

# ==================================================
# TABS
# ==================================================

tab1, tab2, tab3 = st.tabs(
    [
        "📊 Model Overview",
        "🔍 Credit Prediction",
        "📈 Analytics"
    ]
)

# ==================================================
# TAB 1
# MODEL OVERVIEW
# ==================================================

with tab1:

    st.header("Model Performance Comparison")

    st.dataframe(
        results_df,
        use_container_width=True
    )

    best_row = results_df.sort_values(
        by="ROC AUC",
        ascending=False
    ).iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Accuracy",
        f"{best_row['Accuracy']:.2%}"
    )

    c2.metric(
        "Precision",
        f"{best_row['Precision']:.2%}"
    )

    c3.metric(
        "Recall",
        f"{best_row['Recall']:.2%}"
    )

    c4.metric(
        "F1 Score",
        f"{best_row['F1 Score']:.2%}"
    )

    c5.metric(
        "ROC-AUC",
        f"{best_row['ROC AUC']:.2%}"
    )

    st.success(
        f"🏆 Best Model: {best_row['Model']}"
    )

    st.subheader("Project Summary")

    st.write(
        f"""
        Dataset Records: {len(df):,}

        Features Used: {len(X.columns)}

        Target Variable: Creditworthy
        """
    )

# ==================================================
# TAB 2
# CREDIT PREDICTION
# ==================================================

with tab2:

    st.header("Customer Credit Assessment")

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=21,
            max_value=70,
            value=30
        )

        income = st.number_input(
            "Income",
            min_value=0,
            value=50000
        )

        debt = st.number_input(
            "Debt",
            min_value=0,
            value=10000
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0,
            value=50000
        )

    with col2:

        credit_utilization = st.slider(
            "Credit Utilization",
            0.0,
            1.0,
            0.5
        )

        payment_history = st.slider(
            "Payment History",
            0,
            100,
            80
        )

        employment_years = st.number_input(
            "Employment Years",
            min_value=0,
            max_value=40,
            value=5
        )

    with col3:

        num_credit_cards = st.number_input(
            "Number of Credit Cards",
            min_value=0,
            max_value=15,
            value=2
        )

        existing_loans = st.number_input(
            "Existing Loans",
            min_value=0,
            max_value=8,
            value=1
        )

        late_payments = st.number_input(
            "Late Payments",
            min_value=0,
            max_value=20,
            value=0
        )

        savings = st.number_input(
            "Savings",
            min_value=0,
            value=50000
        )

    debt_to_income = debt / (income + 1)

    if st.button("Predict Creditworthiness"):

        user_input = pd.DataFrame(
            [[
                age,
                income,
                debt,
                loan_amount,
                credit_utilization,
                payment_history,
                employment_years,
                num_credit_cards,
                existing_loans,
                late_payments,
                savings,
                debt_to_income
            ]],
            columns=X.columns
        )

        prediction = best_model.predict(
            user_input
        )[0]

        probability = best_model.predict_proba(
            user_input
        )[0][1]

        if prediction == 1:
            st.success(
                "✅ Customer is CREDITWORTHY"
            )
        else:
            st.error(
                "❌ Customer is NOT CREDITWORTHY"
            )

        st.metric(
            "Approval Probability",
            f"{probability*100:.2f}%"
        )

        st.progress(float(probability))

# ==================================================
# TAB 3
# ANALYTICS
# ==================================================

with tab3:

    st.header("Analytics Dashboard")

    st.subheader("Feature Importance")

    importance = best_model.feature_importances_

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        X.columns,
        importance
    )

    ax.set_title(
        "Feature Importance"
    )

    st.pyplot(fig)

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader("Dataset Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )
    