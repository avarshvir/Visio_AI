import tkinter as tk
from tkinter import messagebox
import mysql.connector
from home import HomeWindow


class LoginWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Login")
        self.geometry("300x200")

        tk.Label(self, text="Username:").grid(row=0, column=0, padx=10, pady=5)
        self.username_entry = tk.Entry(self)
        self.username_entry.grid(row=0, column=1)

        tk.Label(self, text="Password:").grid(row=1, column=0, padx=10, pady=5)
        self.password_entry = tk.Entry(self, show="*")
        self.password_entry.grid(row=1, column=1)

        self.login_button = tk.Button(self, text="Login", command=self.validate_login)
        self.login_button.grid(row=2, column=0, columnspan=2, pady=10)

    def validate_login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()



        # Connecting to MySQL database
        try:
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",
                database="user_db"
            )
            cursor = conn.cursor()

            query = "SELECT * FROM users WHERE email=%s AND password=%s"
            cursor.execute(query, (username, password))
            result = cursor.fetchone()

            if result:
                messagebox.showinfo("Success", "Login Successful!")
                self.destroy()
                home_window = HomeWindow(self.master)
                home_window.mainloop()
            else:
                messagebox.showerror("Error", "Invalid Username or Password!")

        except mysql.connector.Error as err:
            messagebox.showerror("Database Error", f"Error: {err}")

        finally:
            cursor.close()
            conn.close()

