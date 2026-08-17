import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv("spam.csv", encoding="latin-1")

# Handle the original SMS Spam Collection CSV format
if "v1" in df.columns and "v2" in df.columns:
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]

# Handle already cleaned format
elif "label" in df.columns and "message" in df.columns:
    df = df[["label", "message"]]

else:
    raise ValueError("Dataset must contain either v1/v2 or label/message columns.")


df.dropna(inplace=True)

print("Dataset shape:", df.shape)
print(df.head())


# ==============================
# 2. Encode Labels
# ==============================

label_encoder = LabelEncoder()

df["label_encoded"] = label_encoder.fit_transform(df["label"])

X = df["message"].astype(str)
y = df["label_encoded"]


# ==============================
# 3. Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==============================
# 4. Tokenization
# ==============================

MAX_WORDS = 10000
MAX_LENGTH = 100

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)


# ==============================
# 5. Build BiLSTM Model
# ==============================

model = Sequential([
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=128,
        input_length=MAX_LENGTH
    ),

    Bidirectional(
        LSTM(64, return_sequences=False)
    ),

    Dropout(0.5),

    Dense(32, activation="relu"),

    Dropout(0.3),

    Dense(1, activation="sigmoid")
])


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ==============================
# 6. Train Model
# ==============================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

history = model.fit(
    X_train_pad,
    y_train,
    validation_split=0.20,
    epochs=10,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)


# ==============================
# 7. Prediction
# ==============================

y_probability = model.predict(X_test_pad)

y_pred = (y_probability >= 0.5).astype(int).flatten()


# ==============================
# 8. Evaluation
# ==============================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n==============================")
print("BiLSTM MODEL RESULTS")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ==============================
# 9. Save BiLSTM Model
# ==============================

model.save("bilstm_model.keras")

with open("bilstm_tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

with open("bilstm_label_encoder.pkl", "wb") as file:
    pickle.dump(label_encoder, file)


print("\nBiLSTM model saved successfully!")
print("Files created:")
print("- bilstm_model.keras")
print("- bilstm_tokenizer.pkl")
print("- bilstm_label_encoder.pkl")