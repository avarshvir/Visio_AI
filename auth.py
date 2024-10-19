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

                # Set session state after successful signup
                st.session_state.logged_in = True
                st.session_state.user_email = email

                # Use rerun to refresh the app
                st.rerun()
            except mysql.connector.Error:
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
                # Set session state after successful login
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.success("Login successful!")

                # Use rerun to refresh the app
                st.rerun()
            else:
                st.error("Invalid email or password.")
        except mysql.connector.Error as err:
            st.error(f"Database error: {err}")
        finally:
            cursor.close()
            conn.close()
