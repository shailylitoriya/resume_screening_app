import numpy as np
import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Loading the dataset
current_directory = os.getcwd()
dataset_path = os.path.join(current_directory, "resume_dataset.csv")
df = pd.read_csv(dataset_path)
df.head()

# Extracting features and labels
x_text = df["resume_text"] + " " + df["job_description_text"]
y = df["match_score"]

# Spliting the dataset into training and testing
x_train,x_test, y_train, y_test = train_test_split(x_text,y, test_size=0.2, random_state=42, stratify=y)

# Converting text into numerical features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")  # Use max 5000 features
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
x_test_tfidf = tfidf_vectorizer.transform(x_test)

# Initializing and train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train_tfidf, y_train)

# Model evaluation
y_pred = model.predict(x_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:", classification_report(y_test, y_pred))
print("Confusion Matrix:", confusion_matrix(y_test, y_pred))

# Save trained model and vectorizer
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

print("\n Model and Vectorizer Saved Successfully!")
