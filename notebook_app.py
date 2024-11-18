import streamlit as st
import io
import sys

# Shared local variables dictionary
if 'local_vars' not in st.session_state:
    st.session_state.local_vars = {}

# Function to execute the user's code and store variables in session_state
def run_code(user_code):
    # Capture stdout to display code output in Streamlit
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    try:
        # Execute the user's code in the shared local_vars context
        exec(user_code, globals(), st.session_state.local_vars)  # Use the shared local_vars

        output = redirected_output.getvalue()
        return output
    except Exception as e:
        # Return error if code execution fails
        return f"Error: {e}"
    finally:
        sys.stdout = old_stdout

# Function for rendering multiple code cells
def note_edit():
    st.title("Viz NoteBook")

    # Instructions for the user
    st.write("""
        Inbuilt Code Editor""")

    # List to keep track of the code cells and their outputs
    if "cells" not in st.session_state:
        st.session_state.cells = []

    # Render each code cell
    for i, cell in enumerate(st.session_state.cells):
        st.write(f"### Cell {i + 1}")
        user_code = st.text_area(f"Code for Cell {i + 1}", value=cell["code"], height=10, key=f"code_{i}")

        # Option to run each cell
        run_button = st.button(f"Run Cell {i + 1}", key=f"run_button_{i}")
        if run_button:
            output = run_code(user_code)
            st.session_state.cells[i]["output"] = output
            st.session_state.cells[i]["code"] = user_code  # Save the updated code
            st.subheader(f"Output for Cell {i + 1}:")
            st.text(output)

    # Option to add a new code cell
    if st.button("Add New Code Cell"):
        st.session_state.cells.append({"code": "", "output": ""})

if __name__ == "__main__":
    note_edit()