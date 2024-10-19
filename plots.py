import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import io

def select_plots(df, target_variable):
    st.title("Generate Plots")

    # Ensure target variable exists in the dataframe
    if target_variable not in df.columns:
        st.error(f"Target variable '{target_variable}' not found in the dataset.")
        return

    # Allow user to select independent variables (excluding the target variable)
    independent_vars = st.multiselect("Select independent variables:", [col for col in df.columns if col != target_variable])

    if len(independent_vars) == 0:
        st.warning("Please select at least one independent variable.")
        return

    # Plot type selection
    plot_type = st.selectbox(
        "Select the plot type:",
        ["Scatter Plot", "Histogram", "Box Plot", "Correlation Heatmap"]
    )

    # Download button container
    plot_download = st.empty()

    # Function to generate plots
    def generate_plots():
        plots = []  # List to hold the generated plots

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
                plt.title(f"Correlation Heatmap ({target_variable} vs Independent Variables)")

            st.pyplot(plt)

            # Save each plot to a list
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plots.append(buf)

        # Provide download button for the plots
        if plots:
            download_button = plot_download.download_button(
                label="Download Plot(s)",
                data=plots[0],  # Single download for now (can extend for multiple)
                file_name=f"{plot_type}.png",
                mime="image/png"
            )

    # Generate plots button
    if st.button("Generate Plots"):
        generate_plots()
