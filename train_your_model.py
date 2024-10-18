import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

def train_your_model(updated_df, target_variable, train_size = 0.8, random_state=0):
    # Step 1: Preprocess the DataFrame
    # Check if target_variable exists in DataFrame
    if target_variable not in updated_df.columns:
        print(f"Error: '{target_variable}' not found in the DataFrame.")
        return None, None, None  
    
    # Step 2: Split the DataFrame into features and target variable
    X = updated_df.drop(columns=[target_variable])  # Features
    y = updated_df[target_variable]  # Target variable

    # Step 3: Handle categorical features (if any)
    X = pd.get_dummies(X, drop_first=True)  # One-Hot Encoding for categorical features

    # Step 3: Handle categorical features (if any)
    '''for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
    '''
    # Step 4: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=random_state)

    # Step 5: Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

     # Step 6: Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2  # Return the model and metrics

    # Step 6: Train a Linear Regression model (you can choose any model)
   # model = LinearRegression()
   # model.fit(X_train, y_train)

    # Step 7: Make predictions and evaluate the model
    #predictions = model.predict(X_test)
    
    # Calculate evaluation metrics
   # mse = mean_squared_error(y_test, predictions)
   # r2 = r2_score(y_test, predictions)
    
    # Print evaluation results
   # print(f"Mean Squared Error: {mse:.2f}")
#print(f"R² Score: {r2:.2f}")
    
   # return model  # Return the trained model if needed
   

# Example of how to call the function
# Assuming you have the updated_df and target_variable from home.py
# train_your_model(updated_df, 'your_target_column')
