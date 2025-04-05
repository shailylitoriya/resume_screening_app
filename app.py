import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import PyPDF2
import docx

# Extracting text from resume
def extract_text_from_resume(file):
    text = ""
    
    if file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() or "" 
    
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + " "
    
    return text


# Cleaning and preprocessing text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


import numpy as np
import pickle
import os
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
x_train,x_test, y_train, y_test = train_test_split(x_text, y, test_size=0.2, random_state=42, stratify=y)

# Converting text into numerical features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
x_test_tfidf = tfidf_vectorizer.transform(x_test)

# Initializing and training Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train_tfidf, y_train)

# Model evaluation
y_pred = model.predict(x_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save trained model and vectorizer
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

print("\n Model and Vectorizer Saved Successfully!")

# Predicting match score using ML model
def predict_match(resume_text, job_desc_text):
    input_text = [resume_text + " " + job_desc_text]
    input_features = tfidf_vectorizer.transform(input_text)
    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][1] * 100
    return prediction, round(probability, 2)
    

# Streamlit UI
st.title("Resume Matcher App")
st.write("Upload your resume and job description to check the match score!")

# Upload resume
resume_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
job_desc_text = st.text_area("Paste Job Description Here")

if resume_file and job_desc_text:
    with st.spinner("Processing..."):
        # Extracting and cleaning text
        resume_text = extract_text_from_resume(resume_file)
        resume_text = preprocess_text(resume_text)
        job_desc_text = preprocess_text(job_desc_text)

        # Model prediction
        prediction, match_probability = predict_match(resume_text, job_desc_text)

        # Displaying results
        st.success(f"Match Probability: {match_probability}%")
        
        if prediction == 1:
            st.balloons()
            st.write("Strong Match! Your resume is a good fit for this job.")
        else:
            st.warning("Weak Match. Consider optimizing your resume with more relevant skills and keywords.")
