import streamlit as st
import os
import pandas as pd

# Define the path for storing notes
notes_path = "notes"

# Function to ensure notes directory exists
def ensure_notes_dir():
    if not os.path.exists(notes_path):
        os.makedirs(notes_path)

# Function to get notes for the user
def get_notes(email):
    file_path = os.path.join(notes_path, f"{email}.txt")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return f.read()
    return ""

# Function to save notes for the user
def save_notes(email, content):
    file_path = os.path.join(notes_path, f"{email}.txt")
    with open(file_path, "w") as f:
        f.write(content)

# Main Notepad function
def notepad_1():
    st.title("📝 Notepad")
    
    # Get the email from session state
    email = st.session_state.email if "email" in st.session_state else "guest@example.com"

    ensure_notes_dir()

    # Load existing notes for the user
    notes = get_notes(email)
    content = st.text_area("Your Notes:", value=notes, height=400)

    # Save notes button
    if st.button("Save Notes"):
        save_notes(email, content)
        st.success("✅ Notes saved successfully!")

# Run the Notepad function
if __name__ == "__main__":
    notepad_1()
