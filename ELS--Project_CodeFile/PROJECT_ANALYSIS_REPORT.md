# Project Analysis Report — AI-Driven Emotion Detection & Personalized Learning Support Platform

## 1. Architecture Overview
- **Input**: Free-text description of a student's study challenge, plus a selected field of study.
- **Preprocessing**: `data_prep.py` cleans text (lowercasing, URL stripping, punctuation removal).
- **Classification**: Two independent models score the same 5-way emotion space (Bored, Confident, Confused, Curious, Frustrated):
  - **BiLSTM** — trained from scratch, Keras Tokenizer (30k vocab), 80-token sequences, embedding + 2x Bidirectional LSTM + dense head, trained with focal loss.
  - **BERT** — `bert-base-uncased` fine-tuned end-to-end with AdamW (lr=2e-5, 3 epochs), class-weighted cross-entropy loss.
- **Rule layer**: `keyword_rules.py` adds a 10x-weighted keyword boost on top of model probabilities, then renormalizes — catches short, explicit emotional cues the models sometimes miss.
- **Mixed-emotion detection**: any emotion scoring ≥15% is reported alongside the primary one.
- **Guidance generation**: Gemini (`gemini-flash-latest`) generates a field-aware, three-part response (Encouragement / Next Steps / Tip). Falls back to a CSV-backed template (`data/emotion_response_examples.csv`) when Gemini is unavailable.
- **Persistence**: every interaction is appended to `logs/interactions.csv`, powering the Streamlit analytics dashboard (emotion distribution, trend over time, model usage).

## 2. Model Results

| Model | Dataset | Metric | Value |
|---|---|---|---|
| BiLSTM (base, focal loss) | merged GoEmotions + EmpatheticDialogues + ISEAR (198,476 rows) | Train accuracy | 93.97% |
| BiLSTM (base) | same | Val accuracy | 83.39% |
| BiLSTM (base) | same | Training time | 2.42 min |
| BiLSTM (domain-adaptive fine-tune) | 10,000 student-language samples | Val accuracy | ~100%* |
| BERT (bert-base-uncased) | merged dataset | Test accuracy | 95% |
| BERT | same | Best class (Curious) | 1.00 precision / 1.00 recall |
| BERT | same | Confused class | 0.99 precision / 0.93 recall |

\* *The domain-adaptive fine-tune reaching ~100% reflects a small, templated student-language dataset — treat this as confirmation the model adapts to that vocabulary, not as a generalization benchmark. Real-world student input will be noisier than the fine-tuning set.*

## 3. Known Limitations
- **"Bored" is the weakest-covered class.** None of the three source datasets (GoEmotions, EmpatheticDialogues, ISEAR) natively label boredom; it was recovered via keyword-based weak supervision, so it has fewer and noisier examples than the other four classes.
- **Domain-adaptive accuracy is inflated.** 100% val accuracy on a templated student-language set indicates memorization of that specific phrasing pattern, not necessarily robustness to arbitrary real student phrasing.
- **Gemini fallback quality** depends on the CSV template coverage — currently covers General, Computer Science/Programming, and Mathematics; other fields fall back to the General template.
- **Single-language, English-only.** No support for code-mixed or non-English input.

## 4. Cross-Browser / Performance Notes
- Tested in: Chrome, Edge, Firefox *(fill in after your manual pass)*.
- Model loading is cached via `@st.cache_resource`, so cold start is the only slow load; subsequent predictions are near-instant for BiLSTM and low-latency for BERT on CPU.
- Gemini calls typically add 1-3 seconds per request; the fallback path (no API key or call failure) is instant.

## 5. Future Improvements
- Source a dedicated boredom-labeled dataset instead of relying on keyword weak-supervision.
- Expand the field-aware fallback CSV to cover all `STUDY_FIELDS` options.
- Add confidence calibration (e.g. temperature scaling) since raw softmax outputs from small fine-tunes tend to be overconfident.
- A/B test BiLSTM vs BERT vs ensemble on a held-out set of real (not templated) student messages.
