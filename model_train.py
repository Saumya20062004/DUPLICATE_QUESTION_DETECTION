import numpy as np
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, concatenate
from tensorflow.keras.optimizers import Adam

# Load preprocessed data
data = np.load("processed_data.npz")
q1_train = data["q1_train"]
q2_train = data["q2_train"]
q1_val = data["q1_val"]
q2_val = data["q2_val"]
y_train = data["y_train"]
y_val = data["y_val"]

# Load vocabulary
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# Hyperparameters
vocab_size = len(vocab) + 1  # +1 for padding (index 0)
MAX_LEN = q1_train.shape[1]
embedding_dim = 128

# Model architecture
input1 = Input(shape=(MAX_LEN,))
input2 = Input(shape=(MAX_LEN,))

embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=MAX_LEN)
lstm_layer = LSTM(64)

encoded1 = lstm_layer(embedding_layer(input1))
encoded2 = lstm_layer(embedding_layer(input2))

merged = concatenate([encoded1, encoded2])
dense1 = Dense(64, activation='relu')(merged)
output = Dense(1, activation='sigmoid')(dense1)

model = Model(inputs=[input1, input2], outputs=output)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    [q1_train, q2_train], y_train,
    batch_size=128,
    epochs=5,
    validation_data=([q1_val, q2_val], y_val)
)

# Save the trained model
model.save("duplicate_model.h5")
print("✅ Model trained and saved as 'duplicate_model.h5'")
