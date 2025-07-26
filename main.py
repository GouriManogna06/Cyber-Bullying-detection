import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
import spacy
from preprocess import preprocess
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import LambdaCallback

# Optional: HuggingFace BERT in future
# from transformers import BertTokenizer, TFBertModel

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Data loading and cleaning
df = pd.read_csv("cyberbullying_tweets.csv")
df['clean_text'] = df['tweet_text'].apply(preprocess)

X = df['clean_text']
y = df['cyberbullying_type']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)
class_names = label_encoder.classes_

# Tokenization
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
padded_sequences = pad_sequences(sequences, maxlen=100)

X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, y_categorical, test_size=0.2, random_state=42
)

# Load GloVe Embeddings
embedding_index = {}
with open("glove.6B.100d.txt", encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coeffs

embedding_dim = 100
embedding_matrix = np.zeros((5000, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < 5000:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Build model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=embedding_dim,
                    weights=[embedding_matrix], trainable=False))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(32))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
flush_output = LambdaCallback(on_epoch_end=lambda epoch, logs: sys.stdout.flush())
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=5,
          batch_size=64,
          verbose=2,
          callbacks=[flush_output])

# ✅ Save model using native format (no warning)
model.save("cyberbullying_lstm_glove.keras")

# Evaluate
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ROC-AUC
roc_auc_macro = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
print(f"\nROC-AUC (macro): {roc_auc_macro:.4f}")
for i, class_name in enumerate(class_names):
    class_auc = roc_auc_score(y_test[:, i], y_pred_proba[:, i])
    print(f"ROC-AUC for class '{class_name}': {class_auc:.4f}")
