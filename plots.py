import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def select_plots(df, target_variable):
    st.title("Generate Plots")

    independent_vars = [col for col in df.columns if col != target_variable]

    plot_type = st.selectbox("Select the plot type:", ["Scatter Plot", "Histogram", "Box Plot", "Correlation Heatmap"])

    if st.button("Generate Plots"):
        generate_plots(df, target_variable, independent_vars, plot_type)

def generate_plots(df, target_variable, independent_vars, plot_type):
    plots_container = st.container()

    for feature in independent_vars:
        plt.figure(figsize=(10, 6))

        if plot_type == "Scatter Plot":
            sns.scatterplot(x=df[feature], y=df[target_variable])
            plt.title(f"Scatter Plot: {feature} vs {target_variable}")
            plt.xlabel(feature)
            plt.ylabel(target_variable)

        elif plot_type == "Histogram":
            sns.histplot(df[feature], kde=True)
            plt.title(f"Histogram of {feature}")
            plt.xlabel(feature)

        elif plot_type == "Box Plot":
            sns.boxplot(x=df[feature], y=df[target_variable])
            plt.title(f"Box Plot: {feature} vs {target_variable}")
            plt.xlabel(feature)
            plt.ylabel(target_variable)

        elif plot_type == "Correlation Heatmap":
            corr_matrix = df[independent_vars + [target_variable]].corr()
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title(f"Correlation Heatmap")
            break

        with plots_container:
            st.pyplot(plt)

        plt.clf()
