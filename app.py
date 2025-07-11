import streamlit as st
import numpy as np
import pickle
import re
import nltk
import os
import gdown  # <--- NEW

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 📥 Download model from Google Drive if not exists
if not os.path.exists("duplicate_model.h5"):
    url = "https://drive.google.com/file/d/1Ch-SQz4pNkfSxbFNhUYsCte900589-5Q/view?usp=sharing"
    gdown.download(url, "duplicate_model.h5", quiet=False)

# 📥 Similarly download vocab.pkl if stored in Drive (optional)
# if not os.path.exists("vocab.pkl"):
#     vocab_url = "YOUR_VOCAB_PICKLE_FILE_LINK"
#     gdown.download(vocab_url, "vocab.pkl", quiet=False)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Constants
MAX_LEN = 30

# Load model and vocabulary
model = load_model("duplicate_model.h5")
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

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
        q1_input = preprocess_question(q1)
        q2_input = preprocess_question(q2)
        prediction = model.predict([q1_input, q2_input])[0][0]

        if prediction >= 0.5:
            st.success(f"✅ Duplicate (Confidence: {prediction:.2f})")
        else:
            st.error(f"❌ Not Duplicate (Confidence: {1 - prediction:.2f})")
