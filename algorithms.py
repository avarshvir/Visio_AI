"""
import streamlit as st
from sklearn.linear_model import LinearRegression

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
    if target_type == "classification":
        classifier_choice = st.selectbox("Choose a classification algorithm", 
                                         ["Decision Tree", "K-Nearest Neighbors", "Logistic Regression"], 
                                         key="classifier_selectbox")
        # Add additional algorithm selection with unique keys if needed
    elif target_type == "regression":
        regression_choice = st.selectbox("Choose a regression algorithm", 
                                         ["Linear Regression", "Decision Tree Regressor", "KNN Regressor"], 
                                         key="regressor_selectbox")
    else:
        algorithms = regression_algorithms + classification_algorithms
        st.warning("⚠️ Please select a target variable first for specialized algorithms")

    selected_algorithm = st.selectbox("Choose an algorithm", algorithms)

    #if st.button("Select Algorithm"):
    #    st.success(f"{selected_algorithm} selected!")
    #    return selected_algorithm
    selected_algorithm = st.selectbox("Choose an algorithm", algorithms)
    if st.button("Select Algorithm"):
        if 'trained_model' in st.session_state:
            # Load the test dataset
            X_test = st.session_state['trained_model']['X_test']
            y_test = st.session_state['trained_model']['y_test']
            
            # Initialize and fit the model (replace with selected algorithm)
            model = LinearRegression() if selected_algorithm == "Linear Regression" else None
            model.fit(st.session_state['trained_model']['X_train'], st.session_state['trained_model']['y_train'])
            
            # Generate predictions
            predictions = model.predict(X_test)
            st.session_state['predictions'] = predictions  # Store predictions in session state

            st.success(f"{selected_algorithm} selected and predictions generated!")
    
    return None"""
#algorithms.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
#from xgboost import XGBRegressor, XGBClassifier

def select_algorithms(target_type=None):
    st.subheader("🧩 Select Algorithms")
    
    # Check if model is trained first
    if 'trained_model' not in st.session_state:
        st.error("❌ Please train your model first before selecting algorithms!")
        return None
        
    # Get target type from trained model
    target_type = st.session_state['trained_model'].get('target_type')
    
    if target_type == 'numerical':
        algorithms = {
            "Linear Regression": LinearRegression,
            "Ridge Regression": Ridge,
            "Lasso Regression": Lasso,
            "Random Forest Regressor": RandomForestRegressor,
            "SVR (Support Vector Regression)": SVR,
  #          "XGBoost Regressor": XGBRegressor
        }
    else:  # categorical
        algorithms = {
            "Logistic Regression": LogisticRegression,
            "Random Forest Classifier": RandomForestClassifier,
            "Decision Tree Classifier": DecisionTreeClassifier,
            "SVM Classifier": SVC,
            "KNN Classifier": KNeighborsClassifier,
 #           "XGBoost Classifier": XGBClassifier
        }

    # Algorithm selection
    selected_algorithm = st.selectbox(
        "Choose an algorithm",
        list(algorithms.keys()),
        key="algorithm_selectbox"
    )

    # Only show the Select Algorithm button if we have training data
    if st.button("Train Selected Algorithm"):
        if 'trained_model' in st.session_state:
            try:
                # Get training and test data
                X_train = st.session_state['trained_model']['X_train']
                X_test = st.session_state['trained_model']['X_test']
                y_train = st.session_state['trained_model']['y_train']
                y_test = st.session_state['trained_model']['y_test']
                target_variable = st.session_state['trained_model']['target_variable']
                
                # Initialize the selected algorithm
                model = algorithms[selected_algorithm]()
                
                # Show training progress
                with st.spinner(f"Training {selected_algorithm}..."):
                    # Fit the model
                    model.fit(X_train, y_train)
                    
                    # Generate predictions
                    predictions = model.predict(X_test)
                    
                    # Store predictions and model in session state
                    st.session_state['predictions'] = predictions
                    st.session_state['current_model'] = model
                    
                    # Calculate metrics
                    if target_type == 'numerical':
                        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                        mse = mean_squared_error(y_test, predictions)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, predictions)
                        r2 = r2_score(y_test, predictions)
                        metrics = {
                            "Mean Squared Error": float(mse),
                            "Root Mean Squared Error": float(rmse),
                            "Mean Absolute Error": float(mae),
                            "R² Score": float(r2)
                        }
                    else:
                        from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
                        accuracy = accuracy_score(y_test, predictions)
                        precision = precision_score(y_test, predictions, average='weighted')
                        recall = recall_score(y_test, predictions, average='weighted')
                        metrics = {
                            "Accuracy": float(accuracy),
                            "Precision": float(precision),
                            "Recall": float(recall)
                        }
                    
                    st.session_state['evaluation_metrics'] = metrics
                
                st.success(f"✅ {selected_algorithm} trained successfully!")
                
                # Create predictions DataFrame
                predictions_df = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': predictions,
                    'Difference': y_test - predictions if target_type == 'numerical' else None
                })
                
                # Display results in an expander
                with st.expander("📊 View Predictions and Model Performance", expanded=True):
                    # Display evaluation metrics
                    st.subheader("Model Performance Metrics")
                    metrics_df = pd.DataFrame([metrics]).T
                    metrics_df.columns = ['Value']
                    st.dataframe(metrics_df)
                    
                    # Display predictions
                    st.subheader("Predictions on Test Data")
                    st.dataframe(predictions_df)
                    
                    # Add download button for predictions
                    csv = predictions_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name=f'{selected_algorithm}_predictions.csv',
                        mime='text/csv'
                    )
                    
                    # Display scatter plot for numerical predictions
                    if target_type == 'numerical':
                        st.subheader("Actual vs Predicted Values")
                        import plotly.express as px
                        fig = px.scatter(
                            predictions_df, x='Actual', y='Predicted',
                            title=f'Actual vs Predicted {target_variable}',
                            labels={'Actual': f'Actual {target_variable}', 
                                   'Predicted': f'Predicted {target_variable}'}
                        )
                        fig.add_scatter(
                            x=[predictions_df['Actual'].min(), predictions_df['Actual'].max()],
                            y=[predictions_df['Actual'].min(), predictions_df['Actual'].max()],
                            mode='lines', name='Perfect Prediction', line=dict(dash='dash')
                        )
                        st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
        else:
            st.error("❌ No training data available. Please train your model first.")
    
    return None