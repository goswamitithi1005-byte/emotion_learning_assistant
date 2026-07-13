"""
Epic 2: Train the BiLSTM emotion classifier and save model + tokenizer + label encoder.

Run:
    python src/train_bilstm.py
"""
import os
import sys
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MAX_SEQUENCE_LENGTH, VOCAB_SIZE, EMBEDDING_DIM, BILSTM_UNITS,
    BATCH_SIZE, EPOCHS, BILSTM_MODEL_PATH, BILSTM_TOKENIZER_PATH,
    BILSTM_LABEL_ENCODER_PATH, EMOTION_LABELS, MODEL_DIR,
)
from data_prep import load_dataset, split_dataset


def build_model(vocab_size, num_classes):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        Bidirectional(LSTM(BILSTM_UNITS, return_sequences=True)),
        Bidirectional(LSTM(BILSTM_UNITS // 2)),
        Dropout(0.4),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_dataset()
    train_df, val_df = split_dataset(df)

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["clean_text"])

    X_train = pad_sequences(tokenizer.texts_to_sequences(train_df["clean_text"]),
                             maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    X_val = pad_sequences(tokenizer.texts_to_sequences(val_df["clean_text"]),
                           maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")

    label_encoder = LabelEncoder()
    label_encoder.fit(EMOTION_LABELS)  # fix class order across the whole project
    y_train = to_categorical(label_encoder.transform(train_df["emotion"]), num_classes=len(EMOTION_LABELS))
    y_val = to_categorical(label_encoder.transform(val_df["emotion"]), num_classes=len(EMOTION_LABELS))

    model = build_model(min(VOCAB_SIZE, len(tokenizer.word_index) + 1), len(EMOTION_LABELS))
    model.summary()

    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
    )

    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Validation accuracy: {val_acc:.4f}")

    model.save(BILSTM_MODEL_PATH)
    with open(BILSTM_TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(BILSTM_LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"Saved model -> {BILSTM_MODEL_PATH}")
    print(f"Saved tokenizer -> {BILSTM_TOKENIZER_PATH}")
    print(f"Saved label encoder -> {BILSTM_LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    main()
