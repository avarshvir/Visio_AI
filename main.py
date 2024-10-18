#auth.py
import streamlit as st
import re
import mysql.connector

# Function for Sign Up
def signup():
    st.title("Sign Up")
    name = st.text_input("Name")
    dob = st.text_input("Date of Birth (YYYY-MM-DD)")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    address = st.text_area("Address")

    if st.button("Submit"):
        if not name or not dob or not email or not password or not address:
            st.error("All fields are required!")
        elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            st.error("Invalid email format!")
        elif len(password) < 6:
            st.error("Password must be at least 6 characters long!")
        else:
            try:
                conn = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="",
                    database="user_db"
                )
                cursor = conn.cursor()
                query = "INSERT INTO users (name, dob, email, password, address) VALUES (%s, %s, %s, %s, %s)"
                values = (name, dob, email, password, address)
                cursor.execute(query, values)
                conn.commit()
                st.success("Account created successfully!")
            except mysql.connector.Error as err:
                st.error("Email is already taken or a database error occurred.")
            finally:
                cursor.close()
                conn.close()

# Function for Login
def login():
    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        try:
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",
                database="user_db"
            )
            cursor = conn.cursor()
            query = "SELECT * FROM users WHERE email=%s AND password=%s"
            cursor.execute(query, (email, password))
            result = cursor.fetchone()

            if result:
                st.session_state.logged_in = True
                st.success("Login successful!")
            else:
                st.error("Invalid email or password.")
        except mysql.connector.Error as err:
            st.error(f"Database error: {err}")
        finally:
            cursor.close()
            conn.close()

# home.py:
import streamlit as st

def home():
    st.title("Home Page")

    # Create two columns: one for tools, one for dataset upload
    col1, col2 = st.columns([1, 3])

    # Column 1: Tools
    with col1:
        st.subheader("Tools")
        st.write("Tool 1: Example Tool")
        st.write("Tool 2: Another Tool")
        st.write("Tool 3: Yet Another Tool")
        st.write("Tool 4: Coming Soon")
        st.write("Tool 5: Coming Soon")

    # Column 2: Dataset Upload
    with col2:
        st.subheader("Upload Your Dataset")
        dataset = st.file_uploader("Choose a dataset file", type=["csv", "xlsx", "txt"])

        if dataset:
            st.success("File uploaded successfully!")
            st.write(f"File name: {dataset.name}")
            # You can process the uploaded dataset here
        else:
            st.info("Please upload a dataset to proceed.")

# app.py:
import streamlit as st
from auth import signup, login  # Import the signup and login from the auth file
from home import home  # Import the home page from home.py

def main():
    st.title("Welcome to Data Analysis - Prediction")

    # Check if user is logged in via session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # If logged in, show home page, else show Sign Up or Login
    if st.session_state.logged_in:
        home()  # Redirect to home page after login
    else:
        # Display the Sign Up and Login options
        menu = ["Sign Up", "Login"]
        choice = st.selectbox("Choose Action", menu)

        if choice == "Sign Up":
            signup()
        elif choice == "Login":
            login()

# Run the app
if __name__ == "__main__":
    main()