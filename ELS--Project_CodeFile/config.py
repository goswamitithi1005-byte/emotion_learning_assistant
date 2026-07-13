"""
Central configuration for the AI-Driven Emotion Detection & Personalized
Learning Support Platform.
"""
import os

# Standalone Keras 3 + PyTorch backend for the BiLSTM model. Must be set
# before *any* `import keras` happens anywhere (it is, since config.py is
# always the first internal import). This avoids depending on
# tensorflow.keras entirely, sidestepping TF-version deserialization
# mismatches between the Kaggle training env and your local machine, and
# reuses the torch install you already have for BERT.
os.environ["KERAS_BACKEND"] = "torch"

from dotenv import load_dotenv

# Load variables from a .env file in the project root, if present.
load_dotenv()

# ---------- Emotion labels (fixed order used everywhere) ----------
EMOTION_LABELS = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_emotions.csv")       # your Kaggle dataset goes here
SAMPLE_DATA_PATH = os.path.join(DATA_DIR, "sample_emotions.csv")  # small demo set (included)

# NOTE: .keras is the native Keras 3 format (recommended over legacy .h5 for
# cross-environment portability). Make sure the file you download from
# Kaggle actually has this extension/format -- if you only have a .h5,
# either re-save it on Kaggle as model.save("bilstm_emotion_baseline.keras")
# or point this path at whatever you actually have.
BILSTM_MODEL_PATH = os.path.join(MODEL_DIR, "bilstm_emotion_baseline.keras")
BILSTM_TOKENIZER_PATH = os.path.join(MODEL_DIR, "bilstm_tokenizer.pkl")
BILSTM_LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "bilstm_label_encoder.pkl")

BERT_MODEL_DIR = os.path.join(MODEL_DIR, "bert_emotion")
BERT_LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "bert_label_encoder.json")

INTERACTION_LOG_PATH = os.path.join(LOG_DIR, "interactions.csv")
FALLBACK_TEMPLATES_PATH = os.path.join(DATA_DIR, "emotion_response_examples.csv")

# ---------- Study fields offered in the UI ----------
STUDY_FIELDS = [
    "General", "Computer Science / Programming", "Mathematics",
    "Physics", "Chemistry", "Biology", "Engineering", "Other",
]

# ---------- Model hyperparameters ----------
# IMPORTANT: MAX_SEQUENCE_LENGTH must match whatever maxlen you actually
# padded to during Kaggle tokenization. Your Kaggle preprocessing screenshot
# showed padded_sequences.shape == (198476, 80) -- if that's the model you're
# loading, this needs to be 80, not 60. Wrong length won't crash, it'll just
# silently truncate/pad differently than training and degrade predictions.
MAX_SEQUENCE_LENGTH = 60  # <-- confirm this against your Kaggle tokenization cell and fix if needed
VOCAB_SIZE = 20000
EMBEDDING_DIM = 128
BILSTM_UNITS = 128
BATCH_SIZE = 16
EPOCHS = 8

BERT_MODEL_NAME = "bert-base-uncased"
BERT_MAX_LENGTH = 64
BERT_BATCH_SIZE = 8
BERT_EPOCHS = 3

# ---------- Gemini ----------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = "gemini-flash-latest"

# ---------- Mixed-emotion detection ----------
MIXED_EMOTION_THRESHOLD = 0.15  # any emotion with prob >= this is reported as "present" (spec: >=15%)
