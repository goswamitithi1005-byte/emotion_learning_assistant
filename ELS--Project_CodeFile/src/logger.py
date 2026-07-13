"""
Epic 6: User Interaction logging.

Every submission is appended to a CSV so the app can show history and
power the analytics dashboard (emotion trends over time, model usage, etc.)
"""
import os
import sys
import csv
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INTERACTION_LOG_PATH, LOG_DIR

FIELDNAMES = [
    "timestamp", "text", "model_used", "top_emotion", "mixed_emotions",
    "confidence", "guidance",
]


def log_interaction(text, model_used, top_emotion, mixed_emotions, confidence, guidance):
    os.makedirs(LOG_DIR, exist_ok=True)
    file_exists = os.path.exists(INTERACTION_LOG_PATH)
    with open(INTERACTION_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "text": text,
            "model_used": model_used,
            "top_emotion": top_emotion,
            "mixed_emotions": "|".join(mixed_emotions),
            "confidence": round(confidence, 4),
            "guidance": guidance.replace("\n", " "),
        })


def load_interaction_log():
    import pandas as pd
    if not os.path.exists(INTERACTION_LOG_PATH):
        return pd.DataFrame(columns=FIELDNAMES)
    return pd.read_csv(INTERACTION_LOG_PATH)


def clear_log():
    """Deletes the interaction log CSV (used by the app's 'Clear History' button)."""
    if os.path.exists(INTERACTION_LOG_PATH):
        os.remove(INTERACTION_LOG_PATH)
