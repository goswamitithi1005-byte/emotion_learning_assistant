"""
Epic 2: Fine-tune BERT for emotion classification and save model + label encoder.

Run:
    python src/train_bert.py
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    BertTokenizerFast, BertForSequenceClassification,
    Trainer, TrainingArguments,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BERT_MODEL_NAME, BERT_MAX_LENGTH, BERT_BATCH_SIZE, BERT_EPOCHS,
    BERT_MODEL_DIR, BERT_LABEL_ENCODER_PATH, EMOTION_LABELS, MODEL_DIR,
)
from data_prep import load_dataset, split_dataset


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True, max_length=max_length
        )
        self.labels = list(labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


class WeightedTrainer(Trainer):
    """Trainer that applies per-class weights to the loss, so a class with
    fewer examples (e.g. your weakly-labeled 'Bored' rows) isn't drowned out
    by more common classes during training."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_dataset()
    train_df, val_df = split_dataset(df)

    label_encoder = LabelEncoder()
    label_encoder.fit(EMOTION_LABELS)
    y_train = label_encoder.transform(train_df["emotion"])
    y_val = label_encoder.transform(val_df["emotion"])

    # Class weights computed from the TRAIN split only (never from val/test),
    # inverse-proportional to class frequency.
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(EMOTION_LABELS)),
        y=y_train,
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    print("Class weights:", dict(zip(label_encoder.classes_, class_weights.round(3))))

    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
    train_ds = EmotionDataset(train_df["clean_text"], y_train, tokenizer, BERT_MAX_LENGTH)
    val_ds = EmotionDataset(val_df["clean_text"], y_val, tokenizer, BERT_MAX_LENGTH)

    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME, num_labels=len(EMOTION_LABELS)
    )

    args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "bert_checkpoints"),
        num_train_epochs=BERT_EPOCHS,
        per_device_train_batch_size=BERT_BATCH_SIZE,
        per_device_eval_batch_size=BERT_BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=10,
        report_to=[],
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print("Validation metrics:", metrics)

    model.save_pretrained(BERT_MODEL_DIR)
    tokenizer.save_pretrained(BERT_MODEL_DIR)
    # Plain JSON instead of a pickled LabelEncoder: your labels are a fixed
    # list of 5 strings, so this avoids scikit-learn pickle version
    # mismatches between the Kaggle training env and your local env.
    label_json_path = os.path.splitext(BERT_LABEL_ENCODER_PATH)[0] + ".json"
    with open(label_json_path, "w") as f:
        json.dump(list(label_encoder.classes_), f)
    print(f"Saved BERT model -> {BERT_MODEL_DIR}")
    print(f"Saved label list -> {label_json_path}")


if __name__ == "__main__":
    main()
