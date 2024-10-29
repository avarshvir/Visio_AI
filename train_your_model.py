import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def train_your_model(df, target_variable, train_size, random_state):
    # Split the dataset into features (X) and the target variable (y)
    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=random_state)

    # Optionally, train your model here (e.g., a simple regression model)
    # model = SomeModel().fit(X_train, y_train)

    # Return the split data (X_train, X_test, y_train, y_test) and any model performance metrics you'd like to display
    return X_train, X_test, y_train, y_test
