import tkinter as tk
from tkinter import messagebox
import re
import mysql.connector


class SignupWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Sign Up")
        self.geometry("400x300")

        tk.Label(self, text="Name:").grid(row=0, column=0, padx=10, pady=5)
        self.name_entry = tk.Entry(self)
        self.name_entry.grid(row=0, column=1)

        tk.Label(self, text="Date of Birth (YYYY-MM-DD):").grid(row=1, column=0, padx=10, pady=5)
        self.dob_entry = tk.Entry(self)
        self.dob_entry.grid(row=1, column=1)

        tk.Label(self, text="Email:").grid(row=2, column=0, padx=10, pady=5)
        self.email_entry = tk.Entry(self)
        self.email_entry.grid(row=2, column=1)

        tk.Label(self, text="Password:").grid(row=3, column=0, padx=10, pady=5)
        self.password_entry = tk.Entry(self, show="*")
        self.password_entry.grid(row=3, column=1)

        tk.Label(self, text="Address:").grid(row=4, column=0, padx=10, pady=5)
        self.address_entry = tk.Entry(self)
        self.address_entry.grid(row=4, column=1)

        self.submit_button = tk.Button(self, text="Submit", command=self.validate_signup)
        self.submit_button.grid(row=5, column=0, columnspan=2, pady=10)

    def validate_signup(self):
        name = self.name_entry.get()
        dob = self.dob_entry.get()
        email = self.email_entry.get()
        password = self.password_entry.get()
        address = self.address_entry.get()

        if not name or not dob or not email or not password or not address:
            messagebox.showerror("Error", "All fields are required!")
            return

        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            messagebox.showerror("Error", "Invalid email format!")
            return

        if len(password) < 6:
            messagebox.showerror("Error", "Password must be at least 6 characters long!")
            return

        # Connecting to MySQL database
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

            messagebox.showinfo("Success", "Registration Successful!")
            self.destroy()
            self.master.deiconify()

        except mysql.connector.Error as err:
            messagebox.showinfo("Error", "Email is already taken!!")
            #messagebox.showerror("Database Error", f"Error: {err}")

        finally:
            cursor.close()
            conn.close()

