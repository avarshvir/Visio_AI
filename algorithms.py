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
        st.success(f"{selected_algorithm} selected!")  # Update this message based on the selection result
