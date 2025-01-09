import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Function to train the model using the CSV file
def train_model(csv_file):
    try:
        # Load the dataset
        data = pd.read_csv(csv_file)
        
        # Check for missing values and handle them (e.g., drop or fill)
        data.fillna(method='ffill', inplace=True)  # Forward fill for simplicity; adjust as needed
        
        # Assuming the last column is the target variable
        X = data.iloc[:, :-1]  # Features
        y = data.iloc[:, -1]   # Target variable
        
        # Identify categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Create a preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(), categorical_cols)  # One-hot encode categorical features
            ],
            remainder='passthrough'  # Keep other columns as they are
        )
        
        # Create a pipeline that first transforms the data and then fits the model
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier())
        ])
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model_pipeline.fit(X_train, y_train)
        
        # Save the model
        joblib.dump(model_pipeline, 'malware_detection_model.pkl')
        print("[INFO] Model trained and saved successfully.")
        messagebox.showinfo("Success", "Model trained and saved successfully.")
    except Exception as e:
        print(f"[ERROR] Could not train model: {e}")
        messagebox.showerror("Error", f"Could not train model: {e}")

def extract_features(file_path):
    """Extract features from the executable file."""
    return [0] * 10  # Replace with actual feature extraction logic

def is_malware(file_path, model):
    """Check if the file is malware based on the AI model."""
    features = extract_features(file_path)
    try:
        prediction = model.predict([features])
        return prediction[0] == 1  # Assuming 1 indicates malware
    except Exception as e:
        print(f"[ERROR] Prediction error for {file_path}: {e}")
        return False

def scan_directory(directory, model):
    """Scan all files in the given directory for malware."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith('.exe'):  # Check only executable files
                if is_malware(file_path, model):
                    print(f"[ALERT] Malware detected: {file_path}")
                else:
                    print(f"[INFO] No malware detected: {file_path}")

def start_scan():
    """Start the scanning process."""
    model = load_model('malware_detection_model.pkl')
    if model is not None:
        print("[INFO] Starting system scan...")
        scan_directory(os.path.abspath(os.sep), model)  # Start from the root directory
        print("[INFO] Scan complete.")
    else:
        messagebox.showerror("Error", "Model could not be loaded.")

def load_model(model_file):
    """Load the trained model."""
    try:
        model = joblib.load(model_file)
        print("[INFO] Model loaded successfully.")
        return model
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        return None

# Create the GUI
def start_training():
    """Start the training process."""
    csv_file_path = '/home/cybergod/Documents/webapplication firewall/blockchain/Malware dataset.csv'  # Replace with your CSV file path
    train_model(csv_file_path)

# Create the GUI
root = tk.Tk()
root.title("Malware Detection System")

# Load and set the background image
background_image = Image.open("/home/cybergod/Documents/webapplication firewall/blockchain/Screenshot From 2025-01-09 13-48-42.png")  # Replace with the path to your image
#background_image = background_image.resize((557, 522), Image.ANTIALIAS)  # Resize the image to fit the window
bg_image = ImageTk.PhotoImage(background_image)

# Create a label to hold the background image
background_label = tk.Label(root, image=bg_image)
background_label.place(relwidth=1, relheight=1)  # Make the label fill the entire window

# Button to train the model
train_button = tk.Button(root, text="Train Model", command=start_training)
train_button.pack(pady=20)

# Button to scan the system for malware
scan_button = tk.Button(root, text="Scan System", command=start_scan)
scan_button.pack(pady=20)

# Button to exit the application
exit_button = tk.Button(root, text="Exit", command=root.quit)
exit_button.pack(pady=20)

# Start the GUI event loop
root.mainloop()