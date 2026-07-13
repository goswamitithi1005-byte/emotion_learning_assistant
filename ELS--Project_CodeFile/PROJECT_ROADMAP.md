# Project Roadmap — Emotion Detection & Learning Support Engine
### Aligned to the official SkillWallet Epic/Story breakdown

Legend: ✅ Done & verified | 🔧 Needs rework to match spec | ⬜ Not started

---

## Epic 1. Environment Setup and Dependency Configuration — ✅ 100% COMPLETE, VERIFIED
| Story | Status | Action |
|---|---|---|
| 1. Obtain Gemini API Key | ✅ | Verified — live Gemini call succeeded with complete, well-formed response |
| 2. Install Python & Create Virtual Environment | ✅ | Done (Python 3.10, venv) |
| 3. Install Project Dependencies | ✅ | Done, all installed successfully |
| 4. Create the .env Configuration File | ✅ | Done — .env loads GEMINI_API_KEY correctly |
| 5. Verify Model and Data Directories | ✅ | Verified via src/verify_setup.py — all directories and artifacts present |
| 6. Prepare Project Folder Structure | ✅ | Done (src/data/models/logs) |

## Epic 2. Kaggle Model Training and Integration
| Story | Status | Action |
|---|---|---|
| 1. Kaggle Setup (GPU + Dependencies + Data Loading) | ⬜ | Build a Kaggle notebook version of the training pipeline |
| 2. Data Preprocessing & Tokenization | 🔧 | Extend to merge & remap GoEmotions + EmpatheticDialogues + ISEAR |
| 3. BiLSTM Model Training | 🔧 | Retrain on real merged dataset |
| 4. Domain-Adaptive Fine-Tuning | ⬜ | New script: second-pass fine-tuning on student-language examples |
| 5. BERT Model Fine-Tuning | 🔧 | Add class weighting; retrain on real data |
| 6. Model Export & Local Integration | 🔧 | Download Kaggle-trained models into local `models/` folder |

## Epic 3. Core Emotion Detection Pipeline Development
| Story | Status | Action |
|---|---|---|
| 1. Text Preprocessing & Keyword Enhancement | ✅ | Done — reworked to explicit 10x-weight scoring + renormalization |
| 2. BiLSTM Classifier (5-Class Softmax) | ✅ | Already implemented this way |
| 3. BERT Classifier w/ Class Weighting & Keyword Adjustments | 🔧 | Add class weighting (ties to Epic 2 Story 5) |
| 4. Mixed-Emotion Detection (≥15% Secondary Scores) | ✅ | Done — threshold corrected from 0.25 → 0.15 |
| 5. Unified Prediction Schema | ✅ | `EmotionDetector.predict()` already unifies both models |
| 6. CSV Persistence & Cached Model Loading | ✅ | Logging done; `@st.cache_resource` already caches models |

## Epic 4. AI-Powered Guidance & Regeneration Engine
| Story | Status | Action |
|---|---|---|
| 1. Capture Field + Problem, Build Gemini Prompt with Emotion/Confidence | ⬜ | Add a "What field are you studying?" dropdown to the UI |
| 2. Generate Empathetic, Field-Aware Responses; Fallback to Templates | 🔧 | Update prompt to include field; move fallback templates into a CSV |
| 3. Regenerate Responses When Input/AI Toggle Changes; Keep Scores in Sync | ✅ | Already implemented |
| 4. Maintain Session History and CSV Logs for Continuous Learning | ✅ | Already implemented |

## Epic 5. Streamlit UI Implementation
| Story | Status | Action |
|---|---|---|
| 1. Responsive Layout with Session State for Inputs/Results/Analytics | ✅ | Already implemented |
| 2. Sections for field selection, problem input, model comparison, mixed emotions, confidence bars, analytics tabs | 🔧 | Add field selection section |
| 3. Form controls, run/clear, spinners, error handling | 🔧 | Add a "Clear History" button + loading spinner |
| 4. Visualize Scores and Plotly Charts with Caching | ✅ | Already implemented |

## Epic 6. User Interaction
| Story | Status | Action |
|---|---|---|
| 1. Validate UI Flow End-to-End | ✅ | Done — verified live in browser |
| 2. Optimization and Deployment Readiness | ⬜ | Cross-browser check, performance notes, write `PROJECT_ANALYSIS_REPORT.md` |

---

## Execution Order (today + buffer day)

**Session A — Quick wins (Epic 1 + Epic 3 fixes)**
1. Get Gemini API key
2. Add `.env` support + `.env.example`
3. Add directory-verification script
4. Fix mixed-emotion threshold (0.25 → 0.15)
5. Rework keyword rules to "10x weight + renormalize" style

**Session B — Epic 4 + Epic 5 UI work**
6. Add field/domain dropdown to `app.py`
7. Update Gemini prompt to include field
8. Move fallback guidance into `emotion_response_examples.csv` + `emotion_response_mapping.csv`
9. Add sidebar dashboard (Models loaded / Total Interactions / CSV Examples / Clear History)

**Session C — Epic 2 real data + retraining (the big one, do with buffer time)**
10. Source GoEmotions, EmpatheticDialogues, ISEAR datasets
11. Write a merge/remap script → unified 5-label dataset
12. Add class weighting to `train_bert.py`
13. Set up Kaggle notebook (GPU) with the same pipeline
14. Train both models on Kaggle, download results into local `models/`
15. Write the domain-adaptive fine-tuning script (second pass on student-language examples)

**Session D — Epic 6 wrap-up**
16. Cross-browser test (Chrome, Edge, Firefox)
17. Performance check (model load-once confirmed, response time notes)
18. Write `PROJECT_ANALYSIS_REPORT.md`
19. Final full run-through demo

---
*This file lives at the project root — update statuses as we complete each item.*
