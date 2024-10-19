import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from train_your_model import train_your_model
from algorithms import select_algorithms
from plots import select_plots

# Set Streamlit page configuration
st.set_page_config(page_title="Dynamic Data Analysis & Visualization Dashboard", layout="wide")

# Home function for displaying the dashboard
def home():
    # Use session state to store the DataFrame
    if 'updated_df' not in st.session_state:
        st.session_state.updated_df = None  # Initialize updated_df in session state

    st.markdown("<h1 style='text-align: center; color: #4A90E2;'>📊 Dynamic Data Analysis & Visualization Dashboard</h1>", unsafe_allow_html=True)

    # Apply custom styles to beautify the layout
    st.markdown(
        """
        <style>
        .container {
            padding: 1px;
            background-color: #f8f9fa;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
        }
        .section-title {
            font-size: 18px;
            font-weight: bold;
            color: #34495e;
            margin-bottom: 10px;
        }
        .dataset-preview {
            border: 1px solid #d3d3d3;
            border-radius: 15px;
            margin: 20px 0;
            padding: 0.5px;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Create a main container for tools and data handling
    st.markdown('<div class="container">', unsafe_allow_html=True)

    # Button row for navigating to model training, algorithm selection, and plot generation
    col1, col2, col3 = st.columns([1, 1, 1])
    
    """with col1:
        if st.button("🔍 Train Your Model"):
            if st.session_state.updated_df is not None:  # Ensure updated_df is initialized
                # Get the target variable (you can modify this to get the actual target variable from the user)
                target_variable = st.text_input("Enter your target variable:")
                if target_variable:  # Check if target variable is provided
                    model = train_your_model(st.session_state.updated_df, target_variable)  # Call the function to train the model
                    st.success("✅ Model trained successfully!")
                else:
                    st.warning("Please specify a target variable.")
            else:
                st.warning("Please upload a dataset first.")"""
    with col1:
        if st.session_state.updated_df is not None:
            with st.expander("🔍 Train Your Model", expanded=False):
                with st.form(key='train_model_form'):
            #if st.session_state.updated_df is not None:
                    target_variable = st.selectbox("Select the target variable:", st.session_state.updated_df.columns, key="target_variable")

                # Train size and random state inputs
                    train_size = st.slider("Select Train Size (fraction of data for training)", min_value=0.1, max_value=0.9, value=0.8, key="train_size")
                    random_state = st.number_input("Enter Random State (for reproducibility)", value=42, key="random_state")

                # Submit button for the form
                    submit_button = st.form_submit_button(label="Train Model")

                    if submit_button:
                        if target_variable:
                            X_train, X_test, y_train, y_test = train_your_model(st.session_state.updated_df, target_variable, train_size, random_state)

                        # Show previews of the training and testing sets
                            st.write("Training Set Preview:")
                            st.dataframe(X_train)

                            st.write("Test Set Preview:")
                            st.dataframe(X_test)

                            st.success("✅ Model trained successfully!")

        
                
                        else:
                            st.error("Please select a valid target variable.")
                # Call the function to train the model with the specified parameters
                    #model = train_your_model(st.session_state.updated_df, target_variable, train_size, random_state)  
                   
        else:
            st.error("Please upload a dataset first.")
            

    with col2:
        if st.button("⚙️ Select Algorithms"):
            select_algorithms()  # Call the function to select algorithms

    with col3:
        if st.button("📊 Generate Plots"):
            if st.session_state.updated_df is not None:  # Ensure the updated dataset is available
                target_variable = st.selectbox("Select the target variable for plotting:", st.session_state.updated_df.columns)
                select_plots(st.session_state.updated_df, target_variable) 
            #select_plots()  # Call the function to generate plots

            else:
                st.warning("Please upload and process the dataset first.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Column 1: Tools Section (Far Left)
    col1, col_spacer, col2, col_spacer2, col3 = st.columns([1.5, 0.5, 4, 0.5, 4])
    
    with col1:
        st.markdown('<div class="section-title">🛠️ Tools</div>', unsafe_allow_html=True)
        if st.button("🔧 Tool 1: Example Tool"):
            st.session_state.current_page = "notepad_1"  # Set the current page to 'notepad'
            st.rerun()


    # Column 2: Dataset Upload and Handling Section (Center)
    with col2:
        st.markdown('<div class="section-title">📂 Upload Your Dataset</div>', unsafe_allow_html=True)
        dataset = st.file_uploader("Choose a dataset file", type=["csv", "xlsx", "txt"])

        if dataset:
            st.success("✅ File uploaded successfully!")
            st.write(f"File name: **{dataset.name}**")

            # Read the dataset based on file extension
            if dataset.name.endswith(".csv"):
                df = pd.read_csv(dataset)
            elif dataset.name.endswith(".xlsx"):
                df = pd.read_excel(dataset)
            elif dataset.name.endswith(".txt"):
                df = pd.read_csv(dataset, delimiter="\t")

            # Display original dataset in the center
            st.markdown('<div class="dataset-preview">', unsafe_allow_html=True)
            st.subheader("🔍 Original Dataset Preview")
            st.dataframe(df, width=1500)
            st.markdown('</div>', unsafe_allow_html=True)

            # Initialize the updated dataframe for missing value handling
            st.session_state.updated_df = df.copy()  # Store the DataFrame in session state

    # Column 3: Missing Values Handling Section (Far Right)
    with col3:
        if st.session_state.updated_df is not None:  # Ensure updated_df is initialized
            st.markdown('<div class="section-title">📊 Missing Values Report</div>', unsafe_allow_html=True)
            null_counts = st.session_state.updated_df.isnull().sum()
            total_nulls = null_counts.sum()

            if total_nulls == 0:
                st.success("✅ No null values found in the dataset!")
            else:
                st.warning(f"⚠️ Found {total_nulls} null values in the dataset.")
                st.write(null_counts[null_counts > 0])

                # Automatic Missing Value Handling
                st.markdown('<div class="section-title">🤖 Automatic Missing Value Handling</div>', unsafe_allow_html=True)
                default_filling = st.checkbox("Apply default handling (Mean for numerical, Mode for categorical)")

                if default_filling:
                    for col in st.session_state.updated_df.columns:
                        if st.session_state.updated_df[col].isnull().sum() > 0:  # Check for missing values
                            if st.session_state.updated_df[col].dtype == "object":  # Categorical data
                                st.session_state.updated_df[col].fillna(st.session_state.updated_df[col].mode()[0], inplace=True)
                            else:  # Numerical data
                                st.session_state.updated_df[col].fillna(st.session_state.updated_df[col].mean(), inplace=True)
                    st.success("🎉 Missing values have been handled automatically!")

                # Manual Missing Value Handling
                st.markdown('<div class="section-title">🛠️ Manual Missing Value Handling</div>', unsafe_allow_html=True)
                for col in st.session_state.updated_df.columns:
                    if st.session_state.updated_df[col].isnull().sum() > 0:
                        st.write(f"Column: {col} (Missing values: {st.session_state.updated_df[col].isnull().sum()})")
                        if st.session_state.updated_df[col].dtype == "object":  # Categorical data
                            fill_option = st.selectbox(f"Choose a method for {col}", ["Mode", "Fill with a value"])
                            if fill_option == "Fill with a value":
                                fill_value = st.text_input(f"Enter the value to fill for {col}")
                                if st.button(f"Apply to {col}"):
                                    st.session_state.updated_df[col].fillna(fill_value, inplace=True)
                                    st.success(f"Filled {col} missing values with '{fill_value}'!")
                            elif fill_option == "Mode":
                                st.session_state.updated_df[col].fillna(st.session_state.updated_df[col].mode()[0], inplace=True)
                                st.success(f"Filled missing values in {col} using mode!")
                        else:  # Numerical data
                            fill_option = st.selectbox(f"Choose a method for {col}", ["Mean", "Median", "Mode", "Fill with a value"])
                            if fill_option == "Fill with a value":
                                fill_value = st.number_input(f"Enter the value to fill for {col}", value=0.0)
                                if st.button(f"Apply to {col}"):
                                    st.session_state.updated_df[col].fillna(fill_value, inplace=True)
                                    st.success(f"Filled {col} missing values with {fill_value}!")
                            elif fill_option == "Mean":
                                st.session_state.updated_df[col].fillna(st.session_state.updated_df[col].mean(), inplace=True)
                                st.success(f"Filled {col} missing values with mean value!")
                            elif fill_option == "Median":
                                st.session_state.updated_df[col].fillna(st.session_state.updated_df[col].median(), inplace=True)
                                st.success(f"Filled {col} missing values using median!")
                            elif fill_option == "Mode":
                                st.session_state.updated_df[col].fillna(st.session_state.updated_df[col].mode()[0], inplace=True)
                                st.success(f"Filled missing values in {col} using mode!")

            # Display the updated dataset preview below the original dataset in the center column
            with col2:
                st.markdown('<div class="dataset-preview">', unsafe_allow_html=True)
                st.subheader("🔄 Updated Dataset Preview")
                st.dataframe(st.session_state.updated_df, width=1500)
                st.markdown('</div>', unsafe_allow_html=True)

            # Allow user to generate pairplot for updated dataset
            if st.button("📊 Generate Pair Plot"):
                pairplot_data = st.session_state.updated_df.select_dtypes(include=['float64', 'int64'])
                sns.pairplot(pairplot_data)
                plt.title("Pair Plot of Updated Dataset")
                st.pyplot(plt)

# Run the home function
if __name__ == "__main__":
    home()
