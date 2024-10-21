"""import streamlit as st
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
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

def select_plots(df, target_variable):
    st.title("📊 Generate Plots")
    
    # Get numerical columns only
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    independent_vars = [col for col in numerical_cols if col != target_variable]
    
    # Plot selection section
    plot_types = {
        "Scatter Plot": "Shows relationship between two variables",
        "Histogram": "Shows distribution of variables",
        "Box Plot": "Shows statistical summary and outliers",
        "Violin Plot": "Shows probability density of the data",
        "Correlation Heatmap": "Shows correlation between all variables",
        "Pair Plot": "Shows pairwise relationships between variables",
        "Joint Plot": "Shows both distribution and correlation",
        "Regression Plot": "Shows linear regression fit"
    }
    
    # Create two columns for plot selection and preview
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Select Plot Type")
        plot_type = st.selectbox(
            "Choose visualization:",
            options=list(plot_types.keys()),
            format_func=lambda x: f"{x}"
        )
        
        # Show plot description
        st.info(plot_types[plot_type])
        
        # Additional settings based on plot type
        settings = {}
        if plot_type in ["Scatter Plot", "Joint Plot", "Regression Plot"]:
            settings['x_var'] = st.selectbox("Select X variable:", independent_vars)
        elif plot_type == "Correlation Heatmap":
            settings['annot'] = st.checkbox("Show correlation values", value=True)
            settings['cmap'] = st.selectbox("Color scheme:", ["coolwarm", "viridis", "YlOrRd", "RdBu"])
        
        # Generate plot button
        generate_button = st.button("🎨 Generate Plot")
    
    # Plot generation and display
    if generate_button:
        with col2:
            fig = generate_plot(df, target_variable, plot_type, settings)
            st.pyplot(fig)
            
            # Add download button
            download_plot(fig, plot_type)

def generate_plot(df, target_variable, plot_type, settings):
    plt.figure(figsize=(12, 8))
    
    if plot_type == "Scatter Plot":
        sns.scatterplot(data=df, x=settings['x_var'], y=target_variable)
        plt.title(f"Scatter Plot: {settings['x_var']} vs {target_variable}")
    
    elif plot_type == "Histogram":
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sns.histplot(data=df, x=target_variable, kde=True, ax=axes[0])
        axes[0].set_title(f"Distribution of {target_variable}")
        
        # Create histograms for independent variables
        for var in df.select_dtypes(include=['float64', 'int64']).columns:
            if var != target_variable:
                sns.histplot(data=df, x=var, kde=True, ax=axes[1])
                axes[1].set_title(f"Distribution of Independent Variables")
                break
    
    elif plot_type == "Box Plot":
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df_melted = df[numerical_cols].melt()
        sns.boxplot(data=df_melted, x='variable', y='value')
        plt.xticks(rotation=45)
        plt.title("Box Plot of Numerical Variables")
    
    elif plot_type == "Violin Plot":
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df_melted = df[numerical_cols].melt()
        sns.violinplot(data=df_melted, x='variable', y='value')
        plt.xticks(rotation=45)
        plt.title("Violin Plot of Numerical Variables")
    
    elif plot_type == "Correlation Heatmap":
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = df[numerical_cols].corr()
        sns.heatmap(corr_matrix, 
                   annot=settings.get('annot', True),
                   cmap=settings.get('cmap', 'coolwarm'),
                   fmt='.2f')
        plt.title("Correlation Heatmap")
    
    elif plot_type == "Pair Plot":
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        g = sns.pairplot(df[numerical_cols], diag_kind='kde')
        g.fig.suptitle("Pair Plot of Numerical Variables", y=1.02)
        return g.fig
    
    elif plot_type == "Joint Plot":
        g = sns.jointplot(data=df, x=settings['x_var'], y=target_variable, kind='reg')
        g.fig.suptitle(f"Joint Plot: {settings['x_var']} vs {target_variable}", y=1.02)
        return g.fig
    
    elif plot_type == "Regression Plot":
        sns.regplot(data=df, x=settings['x_var'], y=target_variable)
        plt.title(f"Regression Plot: {settings['x_var']} vs {target_variable}")
    
    plt.tight_layout()
    return plt.gcf()

def download_plot(fig, plot_type):
    # Create a bytes buffer for the image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    
    # Create download button
    btn = st.download_button(
        label="📥 Download Plot",
        data=buf,
        file_name=f"{plot_type.lower().replace(' ', '_')}.png",
        mime="image/png"
    )
    
    if btn:
        st.success("✅ Plot downloaded successfully!")"""

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import io
from PIL import Image

"""def select_plots(df, target_variable):
    st.title("Generate Plots")

    independent_vars = [col for col in df.columns if col != target_variable]

    # Allow selecting multiple independent variables for comparison
    selected_vars = st.multiselect("Select independent variables for comparison:", independent_vars)

    # Allow selecting the type of plot
    plot_type = st.selectbox("Select the plot type:", ["Scatter Plot", "Histogram", "Box Plot", "Correlation Heatmap"])

    if st.button("Generate Plots"):
        if selected_vars:
            generate_plots(df, target_variable, selected_vars, plot_type)
        else:
            st.warning("Please select at least one independent variable.")"""

def select_plots(df, target_variable):
    st.title("Generate Plots")
    
    independent_vars = [col for col in df.columns if col != target_variable]

    #plot_type = st.selectbox("Select the plot type:", ["Scatter Plot", "Histogram", "Box Plot", "Correlation Heatmap"], key="plot_type_select")
    plot_type = st.selectbox(f"Select the plot type for {target_variable}:", ["Scatter Plot", "Histogram", "Box Plot", "Correlation Heatmap"], key=f"plot_type_select_{target_variable}")


    if st.button("Generate Plots", key="generate_plots_button"):
        generate_plots(df, target_variable, independent_vars, plot_type)

"""def generate_plots(df, target_variable, independent_vars, plot_type):
    plots_container = st.container()

    # Create a figure for multiple subplots
    fig, axs = plt.subplots(len(independent_vars), 1, figsize=(10, len(independent_vars) * 6))

    for i, feature in enumerate(independent_vars):
        ax = axs[i] if len(independent_vars) > 1 else axs

        if plot_type == "Scatter Plot":
            sns.scatterplot(x=df[feature], y=df[target_variable], ax=ax)
            ax.set_title(f"Scatter Plot: {feature} vs {target_variable}")
            ax.set_xlabel(feature)
            ax.set_ylabel(target_variable)

        elif plot_type == "Histogram":
            sns.histplot(df[feature], kde=True, ax=ax)
            ax.set_title(f"Histogram of {feature}")
            ax.set_xlabel(feature)

        elif plot_type == "Box Plot":
            sns.boxplot(x=df[feature], y=df[target_variable], ax=ax)
            ax.set_title(f"Box Plot: {feature} vs {target_variable}")
            ax.set_xlabel(feature)
            ax.set_ylabel(target_variable)

        elif plot_type == "Correlation Heatmap":
            corr_matrix = df[independent_vars + [target_variable]].corr()
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title(f"Correlation Heatmap")
            break

    with plots_container:
        st.pyplot(fig)

    # Create a download button for the generated plot
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = Image.open(buf)
    st.download_button(label="Download Plot", data=buf, file_name="generated_plot.png", mime="image/png")

    # Clear the figure to avoid overlapping plots in future runs
    plt.clf()

"""

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