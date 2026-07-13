"""
Epic 2 helper: shared data-loading / cleaning utilities for BiLSTM & BERT training.

Point RAW_DATA_PATH (in config.py) at your downloaded Kaggle dataset.
Expected columns: `text`, `emotion` (values must be one of config.EMOTION_LABELS,
or remapped using EMOTION_MAP below if your Kaggle dataset uses different names).
"""
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_PATH, SAMPLE_DATA_PATH, EMOTION_LABELS

# If your Kaggle dataset uses different label names, map them here.
# Example (GoEmotions / student-emotion Kaggle sets often use different vocab):
EMOTION_MAP = {
    "boredom": "Bored", "bored": "Bored",
    "confidence": "Confident", "confident": "Confident",
    "confusion": "Confused", "confused": "Confused",
    "curiosity": "Curious", "curious": "Curious",
    "frustration": "Frustrated", "frustrated": "Frustrated", "anger": "Frustrated",
}


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(use_sample_if_missing: bool = True) -> pd.DataFrame:
    """Loads the Kaggle dataset if present, otherwise falls back to the bundled sample."""
    path = RAW_DATA_PATH if os.path.exists(RAW_DATA_PATH) else (SAMPLE_DATA_PATH if use_sample_if_missing else None)
    if path is None:
        raise FileNotFoundError(
            f"No dataset found. Place your Kaggle CSV at {RAW_DATA_PATH} "
            f"(columns: text, emotion)."
        )
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "emotion"])
    df["emotion"] = df["emotion"].astype(str).str.strip().apply(
        lambda e: EMOTION_MAP.get(e.lower(), e)
    )
    df = df[df["emotion"].isin(EMOTION_LABELS)]
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {path}")
    print(df["emotion"].value_counts())
    return df


def split_dataset(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42):
    num_classes = df["emotion"].nunique()
    # Stratified split needs at least 1 row per class in the test set.
    min_test_size = num_classes / len(df)
    if test_size < min_test_size:
        test_size = min_test_size
        print(f"Adjusted test_size to {test_size:.2f} to fit {num_classes} classes in a small dataset.")

    try:
        train_df, val_df = train_test_split(
            df, test_size=test_size, stratify=df["emotion"], random_state=seed
        )
    except ValueError:
        # Dataset too small/imbalanced even for the adjusted size -> fall back to a plain split.
        print("Dataset too small for a stratified split; using a plain random split instead.")
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=seed)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


if __name__ == "__main__":
    data = load_dataset()
    tr, va = split_dataset(data)
    print(f"Train: {len(tr)}  Val: {len(va)}")
