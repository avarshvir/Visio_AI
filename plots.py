import streamlit as st

def select_plots():
    st.subheader("📊 Select Plots")

    plots = [
        "Bar Graph",
        "Line Graph",
        "Scatter Plot",
        "Histogram"
    ]

    selected_plot = st.selectbox("Choose a plot type", plots)

    # Placeholder for generating plots (add your plotting logic here)
    if st.button("Generate Plot"):
        st.success(f"{selected_plot} generated!")  # Update this message based on the plot generation result
