import pandas as pd
import numpy as np
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
df = pd.read_csv("train.csv")
df.dropna(subset=['question1', 'question2'], inplace=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return tokens

df['q1_tokens'] = df['question1'].apply(clean_and_tokenize)
df['q2_tokens'] = df['question2'].apply(clean_and_tokenize)

# Build vocab
vocab = {}
index = 1
for tokens in df['q1_tokens'].tolist() + df['q2_tokens'].tolist():
    for token in tokens:
        if token not in vocab:
            vocab[token] = index
            index += 1

def tokens_to_sequence(tokens):
    return [vocab.get(token, 0) for token in tokens]

df['q1_seq'] = df['q1_tokens'].apply(tokens_to_sequence)
df['q2_seq'] = df['q2_tokens'].apply(tokens_to_sequence)

MAX_LEN = 30
q1_pad = pad_sequences(df['q1_seq'], maxlen=MAX_LEN, padding='post')
q2_pad = pad_sequences(df['q2_seq'], maxlen=MAX_LEN, padding='post')
labels = df['is_duplicate'].values

q1_train, q1_val, q2_train, q2_val, y_train, y_val = train_test_split(
    q1_pad, q2_pad, labels, test_size=0.2, random_state=42
)

np.savez("processed_data.npz", q1_train=q1_train, q2_train=q2_train,
         q1_val=q1_val, q2_val=q2_val, y_train=y_train, y_val=y_val)

with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

print("✅ Preprocessing complete.")
