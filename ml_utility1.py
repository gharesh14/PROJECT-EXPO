import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Function to read data from file
def read_data(file):
    if file.type == "text/csv":
        df = pd.read_csv(file)
    elif file.type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        df = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file type")
    return df

# Function to preprocess data
def preprocess_data(df, target_column, scaler_type):
    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Check if there are only numerical or categorical columns
    numerical_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if len(numerical_cols) > 0:
        # Impute missing values for numerical columns (mean imputation)
        num_imputer = SimpleImputer(strategy='mean')
        X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

        # Scale the numerical features based on scaler_type
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid scaler_type. Choose 'standard' or 'minmax'.")

        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    if len(categorical_cols) > 0:
        # Impute missing values for categorical columns (mode imputation)
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

        # One-hot encode categorical features
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
        X_test_encoded = encoder.transform(X_test[categorical_cols])
        X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(categorical_cols))
        X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(categorical_cols))
        
        X_train = pd.concat([X_train.drop(columns=categorical_cols), X_train_encoded], axis=1)
        X_test = pd.concat([X_test.drop(columns=categorical_cols), X_test_encoded], axis=1)
    
    return X_train, X_test, y_train, y_test

# Function to train the model
def train_model(X_train, y_train, model, model_name):
    # Ensure the directory exists
    model_dir = "trained_model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Save the trained model
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = round(accuracy, 2)
    return accuracy
