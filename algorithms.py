"""
import streamlit as st

def select_algorithms():
    st.subheader("🧩 Select Algorithms")
    
    algorithms = [
        "Linear Regression",
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "KNN",
        "K-Means",
        "SVM"
    ]

    selected_algorithm = st.selectbox("Choose an algorithm", algorithms)

    # Placeholder for the selected algorithm's logic (add your algorithm selection logic here)
    if st.button("Select Algorithm"):
        st.success(f"{selected_algorithm} selected!")  # Update this message based on the selection result"""

import streamlit as st

def select_algorithms(target_type=None):
    st.subheader("🧩 Select Algorithms")
    
    regression_algorithms = [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "Random Forest Regressor",
        "SVR (Support Vector Regression)",
        "XGBoost Regressor"
    ]
    
    classification_algorithms = [
        "Logistic Regression",
        "Random Forest Classifier",
        "Decision Tree Classifier",
        "SVM Classifier",
        "KNN Classifier",
        "XGBoost Classifier"
    ]
    
    # Select appropriate algorithms based on target variable type
    if target_type == 'numerical':
        algorithms = regression_algorithms
        st.info("📊 Showing regression algorithms for numerical prediction")
    elif target_type == 'categorical':
        algorithms = classification_algorithms
        st.info("🎯 Showing classification algorithms for categorical prediction")
    else:
        algorithms = regression_algorithms + classification_algorithms
        st.warning("⚠️ Please select a target variable first for specialized algorithms")

    selected_algorithm = st.selectbox("Choose an algorithm", algorithms)

    if st.button("Select Algorithm"):
        st.success(f"{selected_algorithm} selected!")
        return selected_algorithm
    
    return None