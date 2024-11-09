import streamlit as st

def calculator():
    st.markdown("<h2 style='text-align: center;'>🧮 Calculator</h2>", unsafe_allow_html=True)

    # Initialize expression in session state
    if 'expression' not in st.session_state:
        st.session_state.expression = ""

    # Display the current expression
    st.text_input("Expression", value=st.session_state.expression, key='input', disabled=True, label_visibility="collapsed")

    # Function to update the expression
    def update_expression(value):
        st.session_state.expression += value

    # Function to clear the expression
    def clear_expression():
        st.session_state.expression = ""

    # Function to calculate the result
    def calculate_result():
        try:
            result = eval(st.session_state.expression)
            st.session_state.expression = str(result)
        except Exception:
            st.session_state.expression = "Error"

    # Create a grid layout for buttons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("7", key="btn7", use_container_width=True):
            update_expression("7")
        if st.button("4", key="btn4", use_container_width=True):
            update_expression("4")
        if st.button("1", key="btn1", use_container_width=True):
            update_expression("1")
        if st.button("C", key="btnC", use_container_width=True):
            clear_expression()

    with col2:
        if st.button("8", key="btn8", use_container_width=True):
            update_expression("8")
        if st.button("5", key="btn5", use_container_width=True):
            update_expression("5")
        if st.button("2", key="btn2", use_container_width=True):
            update_expression("2")
        if st.button("0", key="btn0", use_container_width=True):
            update_expression("0")

    with col3:
        if st.button("9", key="btn9", use_container_width=True):
            update_expression("9")
        if st.button("6", key="btn6", use_container_width=True):
            update_expression("6")
        if st.button("3", key="btn3", use_container_width=True):
            update_expression("3")
        if st.button(".", key="btnDot", use_container_width=True):
            update_expression(".")

    with col4:
        if st.button("Plus(+)", key="btnPlus", use_container_width=True):
            update_expression("+")
        if st.button("Subtact(-)", key="btnMinus", use_container_width=True):
            update_expression("-")
        if st.button("Multiply(*)", key="btnMultiply", use_container_width=True):
            update_expression("*")
        if st.button("Divide(/)", key="btnDivide", use_container_width=True):
            update_expression("/")
        if st.button("=", key="btnEqual", use_container_width=True):
            calculate_result()

    # Display the result
    st.text_input("Result", value=st.session_state.expression, key='result', disabled=True, label_visibility="collapsed")

    # Add some CSS to center the calculator and limit its width
    st.markdown(
        """
        <style>
        .calculator-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            width: 250px;  /* Set a fixed width for the calculator */
            margin: auto;  /* Center the calculator */
            padding: 10px;
            border: 1px solid #d3d3d3;
            border-radius: 10px;
            box-shadow: 0 0 5px rgba(0, 0, 0, 0.1);
            background-color: #f8f9fa;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Wrap the calculator in a div with the class
    #st.markdown('<div class="calculator-container">', unsafe_allow_html=True)
    #st.markdown('</div>', unsafe_allow_html=True)