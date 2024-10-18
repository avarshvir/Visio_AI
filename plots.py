import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def select_plots(df, target_variable):
    st.title("Generate Plots")

    # Ensure target variable exists in the dataframe
    if target_variable not in df.columns:
        st.error(f"Target variable '{target_variable}' not found in the dataset.")
        return

    # Options for the user to select plot type
    plot_type = st.selectbox(
        "Select the plot type:",
        ["Scatter Plot", "Histogram", "Box Plot", "Correlation Heatmap"]
    )

    if plot_type == "Scatter Plot":
        # Allow user to select a feature for x-axis
        feature = st.selectbox("Select a feature for the X-axis:", [col for col in df.columns if col != target_variable])

        # Generate scatter plot
        if st.button("Generate Scatter Plot"):
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=df[feature], y=df[target_variable])
            plt.title(f"Scatter Plot: {feature} vs {target_variable}")
            plt.xlabel(feature)
            plt.ylabel(target_variable)
            st.pyplot(plt)

    elif plot_type == "Histogram":
        # Generate histogram for the target variable
        if st.button("Generate Histogram"):
            plt.figure(figsize=(10, 6))
            sns.histplot(df[target_variable], kde=True)
            plt.title(f"Histogram of {target_variable}")
            plt.xlabel(target_variable)
            st.pyplot(plt)

    elif plot_type == "Box Plot":
        # Allow user to select a feature for x-axis
        feature = st.selectbox("Select a feature for the X-axis (categorical):", [col for col in df.columns if col != target_variable])

        # Generate box plot
        if st.button("Generate Box Plot"):
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature], y=df[target_variable])
            plt.title(f"Box Plot: {target_variable} vs {feature}")
            plt.xlabel(feature)
            plt.ylabel(target_variable)
            st.pyplot(plt)

    elif plot_type == "Correlation Heatmap":
        # Generate correlation heatmap
        if st.button("Generate Correlation Heatmap"):
            plt.figure(figsize=(12, 8))
            corr_matrix = df.corr()
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Heatmap")
            st.pyplot(plt)
