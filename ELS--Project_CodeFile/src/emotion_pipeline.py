"""
Epic 3: Core Emotion Detection Pipeline.

EmotionDetector loads both trained models and exposes a single `.predict()`
entry point used by the Streamlit app (Epic 5). Supports:
 - single-model inference (BiLSTM or BERT)
 - dual-model side-by-side comparison
 - rule-based keyword boosting
 - mixed-emotion breakdown (multiple emotions reported when close in confidence)
"""
import os
import sys
import json
import pickle
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# config.py sets KERAS_BACKEND=torch before anything else -- must be
# imported before any `import keras` happens, anywhere.
from config import (
    EMOTION_LABELS, MAX_SEQUENCE_LENGTH, BILSTM_MODEL_PATH, BILSTM_TOKENIZER_PATH,
    BILSTM_LABEL_ENCODER_PATH, BERT_MODEL_DIR, BERT_LABEL_ENCODER_PATH,
    BERT_MAX_LENGTH, MIXED_EMOTION_THRESHOLD,
)
from data_prep import clean_text
from keyword_rules import apply_keyword_boost


class _SimpleLabelList:
    """Minimal stand-in for sklearn's LabelEncoder, built from a plain JSON
    list of class names. Avoids pickling LabelEncoder objects across
    scikit-learn versions (Kaggle training env vs local env)."""

    def __init__(self, classes):
        self.classes_ = np.array(classes)

    def inverse_transform(self, indices):
        return self.classes_[np.asarray(indices)]


class _TokenizerCompat:
    """Stand-in for the legacy keras.preprocessing.text.Tokenizer class,
    used purely as a pickle restore target -- Keras 3 removed that class
    entirely (superseded by TextVectorization), so unpickling a tokenizer
    trained under an older Keras just fails outright otherwise.

    Pickle restores plain objects by creating an empty instance and setting
    its __dict__ directly (no __init__ call), so the real word_index,
    num_words, oov_token etc. saved in the .pkl land on this class
    unchanged. We only need to reimplement texts_to_sequences() on top of
    that data, using the same (long-stable) algorithm the original used.
    """

    def texts_to_sequences(self, texts):
        return list(self.texts_to_sequences_generator(texts))

    def texts_to_sequences_generator(self, texts):
        num_words = getattr(self, "num_words", None)
        oov_token = getattr(self, "oov_token", None)
        word_index = self.word_index
        oov_index = word_index.get(oov_token) if oov_token else None
        filters = getattr(self, "filters", "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n")
        lower = getattr(self, "lower", True)
        split = getattr(self, "split", " ")
        translate_map = str.maketrans(filters, " " * len(filters))

        for text in texts:
            if lower:
                text = text.lower()
            text = text.translate(translate_map)
            words = [w for w in text.split(split) if w]
            vect = []
            for w in words:
                i = word_index.get(w)
                if i is not None:
                    if num_words and i >= num_words:
                        if oov_index is not None:
                            vect.append(oov_index)
                    else:
                        vect.append(i)
                elif oov_index is not None:
                    vect.append(oov_index)
            yield vect


class _CompatUnpickler(pickle.Unpickler):
    """Redirects lookups for the (now-deleted) Keras Tokenizer class to our
    local stand-in above, so pickled tokenizers survive the Keras 2 -> 3
    transition without needing to retrain or re-export from Kaggle."""

    def find_class(self, module, name):
        if name == "Tokenizer" and "keras" in module:
            return _TokenizerCompat
        return super().find_class(module, name)


def _load_pickle_compat(path):
    with open(path, "rb") as f:
        return _CompatUnpickler(f).load()


class EmotionDetector:
    def __init__(self, load_bilstm=True, load_bert=True):
        self.bilstm_model = None
        self.bilstm_tokenizer = None
        self.bilstm_label_encoder = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.bert_label_encoder = None

        if load_bilstm and os.path.exists(BILSTM_MODEL_PATH):
            self._load_bilstm()
        if load_bert and os.path.exists(BERT_MODEL_DIR):
            self._load_bert()

    def _load_bilstm(self):
        import keras
        from keras.models import load_model

        self.bilstm_tokenizer = _load_pickle_compat(BILSTM_TOKENIZER_PATH)

        # Explicitly declare all 12 classes directly to prevent unseen label mismatches
        bilstm_12_labels = [
            "sadness", "joy", "love", "anger", "fear", "surprise", 
            "neutral", "disgust", "shame", "guilt", "interest", "boredom"
        ]
        self.bilstm_label_encoder = _SimpleLabelList(bilstm_12_labels)

        try:
            self.bilstm_model = self._load_with_scrubbed_config()
            print("🎉 BiLSTM model successfully loaded after pre-scrubbing file architecture configuration.")
        except Exception as e:
            print(f"Error loading BiLSTM model: {e}")
            raise e
        
    @staticmethod
    def _load_with_scrubbed_config():
        """
        Directly sanitizes the .keras zip file's config.json metadata contents safely via standard 
        JSON object dictionaries, saving a sanitized file to a temp path so Keras can parse it perfectly.
        """
        import io
        import zipfile
        import tempfile
        import keras
        from keras.models import load_model

        if not os.path.exists(BILSTM_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {BILSTM_MODEL_PATH}")

        with open(BILSTM_MODEL_PATH, "rb") as f:
            zip_data = f.read()

        mem_zip = io.BytesIO(zip_data)
        
        with zipfile.ZipFile(mem_zip, "r") as archive:
            file_contents = {name: archive.read(name) for name in archive.namelist()}

        if "config.json" in file_contents:
            config_obj = json.loads(file_contents["config.json"].decode("utf-8"))
            
            if "config" in config_obj and "layers" in config_obj["config"]:
                for layer in config_obj["config"]["layers"]:
                    if "config" in layer:
                        layer["config"].pop("quantization_config", None)
            
            file_contents["config.json"] = json.dumps(config_obj).encode("utf-8")

        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, "sanitized_bilstm_model.keras")

        with zipfile.ZipFile(temp_file_path, "w", compression=zipfile.ZIP_DEFLATED) as clean_archive:
            for name, content in file_contents.items():
                clean_archive.writestr(name, content)
        
        try:
            model = load_model(temp_file_path)
        finally:
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass

        return model

    def _load_bert(self):
        """Restores the BERT initialization loader using the correct DistilBERT class architecture."""
        import torch
        # Switch from Bert to DistilBert classes to match your saved model type
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        
        self.bert_model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
        self.bert_model.eval()
        self.bert_tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_DIR)
        
        json_path = os.path.splitext(BERT_LABEL_ENCODER_PATH)[0] + ".json"
        if os.path.exists(json_path):
            with open(json_path) as f:
                self.bert_label_encoder = _SimpleLabelList(json.load(f))
        else:
            with open(BERT_LABEL_ENCODER_PATH, "rb") as f:
                self.bert_label_encoder = pickle.load(f)

    # ---------- individual model inference ----------
    def _predict_bilstm(self, text: str) -> dict:
        try:
            from keras.utils import pad_sequences
        except ImportError:
            from tensorflow.keras.preprocessing.sequence import pad_sequences

        seq = self.bilstm_tokenizer.texts_to_sequences([clean_text(text)])
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
        probs = self.bilstm_model.predict(padded, verbose=0)[0]
        labels = self.bilstm_label_encoder.inverse_transform(np.arange(len(probs)))
        return {label: float(p) for label, p in zip(labels, probs)}

    def _predict_bert(self, text: str) -> dict:
        import torch
        inputs = self.bert_tokenizer(
            clean_text(text), return_tensors="pt", truncation=True,
            padding=True, max_length=BERT_MAX_LENGTH,
        )
        with torch.no_grad():
            logits = self.bert_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).numpy()[0]
        labels = self.bert_label_encoder.inverse_transform(np.arange(len(probs)))
        return {label: float(p) for label, p in zip(labels, probs)}

    # ---------- public API ----------
    def predict(self, text: str, model_choice: str = "bilstm", use_keyword_rules: bool = True) -> dict:
        """
        model_choice: "bilstm" | "bert" | "both"
        Returns:
            {
              "bilstm": {emotion: prob, ...}  (if requested)
              "bert": {emotion: prob, ...}    (if requested)
              "final": {emotion: prob, ...}   (the one used for the app: keyword-boosted)
              "top_emotion": str,
              "mixed_emotions": [emotion, ...]  # all emotions above threshold
            }
        """
        result = {}

        if model_choice in ("bilstm", "both") and self.bilstm_model is not None:
            result["bilstm"] = self._predict_bilstm(text)
        if model_choice in ("bert", "both") and self.bert_model is not None:
            result["bert"] = self._predict_bert(text)

        if model_choice == "both":
            base = result.get("bert") or result.get("bilstm")
        else:
            base = result.get(model_choice)

        if base is None:
            raise RuntimeError(
                f"No trained model available for '{model_choice}'. "
                f"Train the models first (Epic 2) or check MODEL_DIR."
            )

        final = apply_keyword_boost(text, base) if use_keyword_rules else base
        result["final"] = final
        result["top_emotion"] = max(final, key=final.get)
        result["mixed_emotions"] = [e for e, p in final.items() if p >= MIXED_EMOTION_THRESHOLD]
        return result


if __name__ == "__main__":
    detector = EmotionDetector()
    out = detector.predict("I'm so lost on recursion, nothing makes sense", model_choice="bilstm")
    print(out)