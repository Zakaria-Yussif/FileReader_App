import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
import re

# File paths (adjust as necessary)
import os
from django.conf import settings
file_path = r"C:\Users\zakar\.cache\kagglehub\datasets\itachi9604\disease-symptom-description-dataset\versions\2\dataset.csv"
file_path_description = r"C:\Users\zakar\.cache\kagglehub\datasets\itachi9604\disease-symptom-description-dataset\versions\2\symptom_Description.csv"
file_path_precaution = r"C:\Users\zakar\.cache\kagglehub\datasets\itachi9604\disease-symptom-description-dataset\versions\2\symptom_precaution.csv"





for path in [file_path, file_path_description, file_path_precaution]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")

def disease_model(data_input, messageData):
    # Clean user input before extracting symptoms
    cleaned_input = data_input.lower()
    cleaned_input = re.sub(r"\bi have\b|\bi've got\b|\bi'm suffering from\b|\bi feel\b", "", cleaned_input)
    cleaned_input = re.sub(r"\band\b", ",", cleaned_input)  # Replace 'and' with commas
    cleaned_input = re.sub(r"[^\w\s,]", "", cleaned_input)  # Remove punctuation
    cleaned_input = re.sub(r"\s+", " ", cleaned_input).strip()  # Normalize whitespace

    # Process cleaned input
    input_symptoms = [s.strip().replace(" ", "_") for s in cleaned_input.split(",") if s.strip()]

    # Load the main dataset
    df = pd.read_csv(file_path)
    symptom_columns = [col for col in df.columns if col.startswith("Symptom_")]
    df[symptom_columns] = df[symptom_columns].fillna("")

    # Combine symptoms into a list and standardize them
    df['all_symptoms'] = df[symptom_columns].apply(
        lambda row: list(set([s.strip().replace(" ", "_").lower() for s in row if s.strip()])),
        axis=1
    )

    # Encode symptoms
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df['all_symptoms'])

    # Filter user input to match known symptoms
    input_symptoms = [s for s in input_symptoms if s in mlb.classes_]
    print("Extracted valid symptoms:", input_symptoms)

    # Encode diseases
    le = LabelEncoder()
    y = le.fit_transform(df['Disease'])

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train a simple model (for quick testing)
    model = Sequential([
        Input(shape=(X.shape[1],)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(len(le.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.2f}")

    if not input_symptoms:
        print("No valid symptoms found for prediction.")
        return messageData

    # Predict disease
    sample_input = mlb.transform([input_symptoms])
    prediction = model.predict(sample_input)
    predicted_class = le.inverse_transform([np.argmax(prediction)])
    disease_predicted = predicted_class[0]

    # Load descriptions and precautions
    disease_desc = pd.read_csv(file_path_description)
    disease_prec = pd.read_csv(file_path_precaution)

    disease_to_desc = dict(zip(disease_desc['Disease'].str.lower(), disease_desc['Description']))
    precaution_columns = ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
    disease_to_prec = {
        row['Disease'].lower(): [row[col] for col in precaution_columns if pd.notna(row[col])]
        for _, row in disease_prec.iterrows()
    }

    # Build the response message
    disease_predicted_lower = disease_predicted.lower()
    description = disease_to_desc.get(disease_predicted_lower, "No description available.")
    precaution_list = disease_to_prec.get(disease_predicted_lower, ["No precautions listed."])
    precaution_str = ", ".join([p.strip(" _[]''").lower() for p in precaution_list])

    message = {
        "user_input": data_input,
        "predicted": disease_predicted,
        "description": description,
        "precaution": precaution_str
    }

    print(f"Predicted Disease: {disease_predicted}")
    print("Message:", message)

    messageData.append({"disease_message": message})
    return messageData
