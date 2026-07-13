"""
Epic 1, Story 5: Verify Model and Data Directories.

Run this after setup (and again after training) to confirm the project
is wired up correctly before moving to the next epic.

Run:
    python src/verify_setup.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, MODEL_DIR, LOG_DIR, RAW_DATA_PATH, SAMPLE_DATA_PATH,
    BILSTM_MODEL_PATH, BILSTM_TOKENIZER_PATH, BILSTM_LABEL_ENCODER_PATH,
    BERT_MODEL_DIR, BERT_LABEL_ENCODER_PATH, GEMINI_API_KEY,
)


def check(label, path_or_bool, is_path=True):
    ok = os.path.exists(path_or_bool) if is_path else bool(path_or_bool)
    status = "✅" if ok else "❌"
    print(f"{status}  {label}")
    return ok


def main():
    print("=== Directory Structure ===")
    check("data/ directory exists", DATA_DIR)
    check("models/ directory exists", MODEL_DIR)
    check("logs/ directory exists", LOG_DIR)

    print("\n=== Data Files ===")
    has_raw = check("Real dataset (data/raw_emotions.csv)", RAW_DATA_PATH)
    check("Sample dataset (data/sample_emotions.csv)", SAMPLE_DATA_PATH)
    if not has_raw:
        print("   -> Using sample data only. Add your Kaggle dataset for real training.")

    print("\n=== BiLSTM Model Artifacts ===")
    check("BiLSTM model file", BILSTM_MODEL_PATH)
    check("BiLSTM tokenizer", BILSTM_TOKENIZER_PATH)
    check("BiLSTM label encoder", BILSTM_LABEL_ENCODER_PATH)

    print("\n=== BERT Model Artifacts ===")
    check("BERT model directory", BERT_MODEL_DIR)
    check("BERT label encoder", BERT_LABEL_ENCODER_PATH)

    print("\n=== Configuration ===")
    check("GEMINI_API_KEY is set", GEMINI_API_KEY, is_path=False)

    print("\nDone. Fix any ❌ above before proceeding to the next epic.")


if __name__ == "__main__":
    main()
