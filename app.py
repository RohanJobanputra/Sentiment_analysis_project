import streamlit as st
import pickle
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import pandas as pd
import numpy as np
import json, seaborn as sns, matplotlib.pyplot as plt

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load trained model + vectorizer
model = joblib.load("sentiment_model.pkl")         # replace with your model path
vectorizer = joblib.load("tfidf_vectorizer.pkl")   # same vectorizer used during training

st.write("Twitter Sentiment Analysis with AI")
text = st.text_input("Enter your tweet: ")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text): 
    text = text.lower()  
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and w not in string.punctuation]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]  
    return " ".join(tokens)

if st.button("Analyse Sentiment"):
    if text != "":
        cleanedtext = preprocess_text(text)
        tfidf_cleanedtext = vectorizer.transform([cleanedtext]) 
        prediction = model.predict(tfidf_cleanedtext)[0]
        if prediction == 0:
            st.write("The sentiment is **Negative** 😞")
        else:
            st.write("The sentiment is **Positive** 😀")
    else: 
        st.write("Please input your tweet again!")

# ================================
# Load Metrics (from training phase)
# ================================
with open("metrics.json", "r") as f:
    metrics = json.load(f)

cm = np.array(metrics["confusion_matrix"])  # load back as numpy for plotting
cr = metrics["classification_report"]

# Confusion Matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["Negative","Positive"], yticklabels=["Negative","Positive"])
st.pyplot(fig)

# Classification Report
st.subheader("Classification Report")
st.json(cr)   # display nicely as JSON
