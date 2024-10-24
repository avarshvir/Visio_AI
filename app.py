import streamlit as st
from auth import signup, login  # Import the signup and login from the auth file
from home import home  # Import the home page from home.py
from notepad import notepad_1

def main():
    #st.title("Welcome to Project DAVP")
    #st.markdown("---")
    # Initialize session state for login and email tracking if not present
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = None
    
        

    # If logged in, show home page; otherwise, show Sign Up or Login
    if st.session_state.logged_in:
        home()  # Show the home page after successful login
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
