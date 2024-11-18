# algorithms.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

def select_algorithms(target_type=None):
    st.subheader("🧩 Select Algorithms")

    # Ensure that model data is available in session state
    #if 'trained_model' not in st.session_state:
    #    st.error("❌ Please split the dataset first in Data Operations!")
    #    return None
    
    if 'trained_model' not in st.session_state:
        st.error("❌ Trained model data is missing. Please split the dataset first.")
        return
    #else:
    #    st.write("Debug: Trained model data")
    #    st.write(st.session_state['trained_model'])
    #else:
        # Debug: Log the trained model data for verification
    #    st.write("Debug: Trained model data available.")
    #    st.write(st.session_state['trained_model'])

    # Retrieve split data and target info
    trained_data = st.session_state['trained_model']
    target_type = trained_data['target_type']
    X_train = trained_data['X_train']
    X_test = trained_data['X_test']
    y_train = trained_data['y_train']
    y_test = trained_data['y_test']
    #target_type = st.session_state['trained_model']['target_type']
    #target_variable = st.session_state['trained_model']['target_variable']
    #X_train = st.session_state['trained_model']['X_train']
    #X_test = st.session_state['trained_model']['X_test']
    #y_train = st.session_state['trained_model']['y_train']
    #y_test = st.session_state['trained_model']['y_test']

    # Define algorithms for each target type
    algorithms = {
        "numerical": {
            "Linear Regression": LinearRegression,
            "Ridge Regression": Ridge,
            "Lasso Regression": Lasso,
            "Random Forest Regressor": RandomForestRegressor,
            "SVR (Support Vector Regression)": SVR,
        },
        "categorical": {
            "Logistic Regression": LogisticRegression,
            "Random Forest Classifier": RandomForestClassifier,
            "Decision Tree Classifier": DecisionTreeClassifier,
            "SVM Classifier": SVC,
            "KNN Classifier": KNeighborsClassifier,
        }
    }[target_type]

    # Algorithm selection dropdown
    selected_algorithm = st.selectbox("Choose an algorithm", list(algorithms.keys()), key="algorithm_selectbox")
    
    # Train button
    """if st.button("Train Selected Algorithm"):
        if not X_train.empty and not y_train.empty:
            try:
            
                model = algorithms[selected_algorithm]()  # Instantiate the selected algorithm
                with st.spinner(f"Training {selected_algorithm}..."):
                    model.fit(X_train, y_train)  # Train the model
                
                # Save the model to session state
                    st.session_state['current_model'] = {
                        'model': model,
                        'algorithm_name': selected_algorithm
                    }
                    st.success(f"✅ {selected_algorithm} trained successfully!")
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
        else:
            st.error("❌ Training data is missing or invalid. Please split the dataset again.")

    # Test button
    if st.button("Test Selected Algorithm"):
        try:
            if 'current_model' in st.session_state:
                #->model = st.session_state['current_model']
                model = st.session_state['current_model']['model']
                #algorithm_name = st.session_state['current_model']['algorithm_name']

                predictions = model.predict(X_test)
                if predictions.shape[0] != y_test.shape[0]:
                    st.error("❌ Mismatch between predictions and actual test data.")
                    return
                #st.write("Debug: Predictions")
                #st.write(predictions)
                #st.write("Debug: y_test")
                #st.write(y_test)

                metrics = calculate_metrics(target_type, y_test, predictions)

                #->st.session_state['evaluation_metrics'] = metrics
                # Store predictions and metrics
                st.session_state['predictions'] = predictions
                st.session_state['evaluation_metrics'] = metrics
                #display_predictions(predictions, y_test, metrics, target_type, target_variable)
                display_predictions(predictions, y_test, metrics, target_type, trained_data['target_variable'])
            else:
                st.error("❌ No model has been trained yet. Please train a model first.")
        except Exception as e:
            st.error(f"❌ Error during testing: {str(e)}")"""

def display_predictions(predictions, y_test, metrics, target_type, target_variable):
    try:
        with st.expander("📊 View Predictions and Model Performance", expanded=True):
            st.subheader("Model Performance Metrics")
            metrics_df = pd.DataFrame([metrics]).T
            metrics_df.columns = ['Value']
            st.dataframe(metrics_df)

            st.subheader("Predictions on Test Data")
            predictions_df = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
            st.dataframe(predictions_df)

        # Download button for predictions
            csv = predictions_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", data=csv, file_name=f'{target_variable}_predictions.csv', mime='text/csv')
        
        # Plot for numerical predictions
            if target_type == 'numerical':
                import plotly.express as px
                fig = px.scatter(predictions_df, x='Actual', y='Predicted', title=f'Actual vs Predicted {target_variable}')
                fig.add_scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Perfect Prediction')
                st.plotly_chart(fig)
    except Exception as e:
        st.error(f"❌ Error displaying predictions: {str(e)}")

def calculate_metrics(target_type, y_test, predictions):
    try:
        if target_type == 'numerical':
            mse = mean_squared_error(y_test, predictions)
            return {
                "Mean Squared Error": mse,
                "Root Mean Squared Error": np.sqrt(mse),
                "Mean Absolute Error": mean_absolute_error(y_test, predictions),
                "R² Score": r2_score(y_test, predictions)
            }
        else:
            return {
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(y_test, predictions, average='weighted'),
                "Recall": recall_score(y_test, predictions, average='weighted')
            }
    except Exception as e:
        st.error(f"❌ Error calculating metrics: {str(e)}")
        return {}
