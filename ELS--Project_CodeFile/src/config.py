"""
Central configuration for the AI-Driven Emotion Detection & Personalized
Learning Support Platform.
"""
import os
os.environ["KERAS_BACKEND"] = "torch"
import os
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

# Change these paths to look like this:
BILSTM_MODEL_PATH = os.path.join(MODEL_DIR, "bilstm_emotion_baseline.keras")
BILSTM_TOKENIZER_PATH = os.path.join(MODEL_DIR, "bilstm_tokenizer.pkl")
BILSTM_LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "bilstm_label_encoder.pkl")

BERT_MODEL_DIR = os.path.join(MODEL_DIR, "bert_emotion")
BERT_LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "bert_label_encoder.json") # Changed to .json

INTERACTION_LOG_PATH = os.path.join(LOG_DIR, "interactions.csv")
FALLBACK_TEMPLATES_PATH = os.path.join(DATA_DIR, "emotion_response_examples.csv")

# ---------- Study fields offered in the UI ----------
STUDY_FIELDS = [
    "General", "Computer Science / Programming", "Mathematics",
    "Physics", "Chemistry", "Biology", "Engineering", "Other",
]

# ---------- Model hyperparameters ----------
MAX_SEQUENCE_LENGTH = 60
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
