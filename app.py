import streamlit as st
import numpy as np
import pickle
import re
import nltk
import os
import gdown

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 📥 Download model from Google Drive if not exists
@st.cache_resource
def download_model():
    if not os.path.exists("duplicate_model.h5"):
        st.info("Downloading model from Google Drive...")
        file_id = "1Ch-SQz4pNkfSxbFNhUYsCte900589-5Q"
        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            gdown.download(url, "duplicate_model.h5", quiet=False)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            st.stop()
    
    return load_model("duplicate_model.h5")

# 📥 Download vocabulary if needed
@st.cache_resource
def download_vocab():
    # If you have vocab.pkl in Google Drive, add it here
    # For now, we'll create a basic vocab if file doesn't exist
    if not os.path.exists("vocab.pkl"):
        st.warning("vocab.pkl not found. Creating basic vocabulary...")
        # Create a basic vocabulary - you should replace this with your actual vocab
        basic_vocab = {f"word_{i}": i for i in range(1000)}
        with open("vocab.pkl", "wb") as f:
            pickle.dump(basic_vocab, f)
        st.info("Basic vocabulary created. For better results, upload your trained vocab.pkl")
    
    with open("vocab.pkl", "rb") as f:
        return pickle.load(f)

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Initialize everything
download_nltk_data()

# Constants
MAX_LEN = 30

# Load model and vocabulary
model = download_model()
vocab = download_vocab()

# Text preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return tokens

def tokens_to_sequence(tokens):
    return [vocab.get(token, 0) for token in tokens]

def preprocess_question(q):
    tokens = clean_and_tokenize(q)
    seq = tokens_to_sequence(tokens)
    padded = pad_sequences([seq], maxlen=MAX_LEN, padding='post')
    return padded

# Streamlit UI
st.title("🤖 Duplicate Question Detector")
st.write("Enter two questions and the model will predict if they are duplicates.")

q1 = st.text_input("Question 1:")
q2 = st.text_input("Question 2:")

if st.button("Check"):
    if q1.strip() == "" or q2.strip() == "":
        st.warning("⚠️ Please enter both questions.")
    else:
        try:
            q1_input = preprocess_question(q1)
            q2_input = preprocess_question(q2)
            prediction = model.predict([q1_input, q2_input])[0][0]

            if prediction >= 0.5:
                st.success(f"✅ Duplicate (Confidence: {prediction:.2f})")
            else:
                st.error(f"❌ Not Duplicate (Confidence: {1 - prediction:.2f})")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.info("This might be due to missing vocabulary file or model incompatibility.")