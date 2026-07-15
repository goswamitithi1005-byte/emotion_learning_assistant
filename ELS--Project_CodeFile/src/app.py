"""
Epic 5 & 6: Streamlit UI + User Interaction / Analytics Dashboard.

Run:
    streamlit run app.py
"""
import os
os.environ["KERAS_BACKEND"] = "torch"
import sys
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from config import EMOTION_LABELS, BILSTM_MODEL_PATH, BERT_MODEL_DIR, STUDY_FIELDS
from emotion_pipeline import EmotionDetector
from gemini_guidance import generate_guidance
from logger import log_interaction, load_interaction_log, clear_log

st.set_page_config(page_title="AI Learning Assistant", page_icon="🎓", layout="wide")

# ---------- cache the heavy model loading ----------
@st.cache_resource
def get_detector():
    return EmotionDetector(
        load_bilstm=os.path.exists(BILSTM_MODEL_PATH),
        load_bert=os.path.exists(BERT_MODEL_DIR),
    )

detector = get_detector()

# Initialize session state tracking properties cleanly
if "last_result" not in st.session_state:
    st.session_state.last_result = None
    st.session_state.last_text = ""
    st.session_state.last_model = ""
    st.session_state.last_field = "General"
if "current_guidance" not in st.session_state:
    st.session_state.current_guidance = None

with st.sidebar:
    st.subheader("System Status")
    st.metric("BiLSTM loaded", "✅" if detector.bilstm_model is not None else "❌")
    st.metric("BERT loaded", "✅" if detector.bert_model is not None else "❌")
    _log_df_sidebar = load_interaction_log()
    st.metric("Total interactions logged", len(_log_df_sidebar))
    from config import FALLBACK_TEMPLATES_PATH
    _fallback_count = len(pd.read_csv(FALLBACK_TEMPLATES_PATH)) if os.path.exists(FALLBACK_TEMPLATES_PATH) else 0
    st.metric("Fallback CSV examples", _fallback_count)

st.title("🎓 AI-Driven Emotion Detection & Personalized Learning Support")
st.caption("Describe your study challenge. The system detects your emotional state and gives tailored guidance.")

tab_assistant, tab_dashboard = st.tabs(["💬 Assistant", "📊 Analytics Dashboard"])

# =========================================================
# TAB 1: ASSISTANT
# =========================================================
with tab_assistant:
    col_input, col_settings = st.columns([3, 1])

    with col_settings:
        st.subheader("Settings")
        available_models = []
        if os.path.exists(BILSTM_MODEL_PATH):
            available_models.append("bilstm")
        if os.path.exists(BERT_MODEL_DIR):
            available_models.append("bert")
        if len(available_models) == 2:
            available_models.append("both")

        if not available_models:
            st.warning("No trained models found yet. Run `src/train_bilstm.py` and/or `src/train_bert.py` first.")
        model_choice = st.radio(
            "Model", options=available_models or ["bilstm"],
            format_func=lambda x: {"bilstm": "BiLSTM", "bert": "BERT", "both": "Compare Both"}.get(x, x),
        )
        use_rules = st.checkbox("Apply keyword rule boosting", value=True)
        show_ai = st.checkbox("Show AI-generated guidance", value=True)

    with col_input:
        field = st.selectbox("What field are you studying?", STUDY_FIELDS)
        user_text = st.text_area(
            "Describe your study challenge",
            placeholder="e.g. I'm so lost on recursion, nothing makes sense...",
            height=120,
        )
        submit = st.button("Analyze", type="primary")

    # Flag parameters to control API invocation loops
    trigger_generation = False
    regenerate = False

    if submit and user_text.strip():
        with st.spinner("Analyzing emotional state..."):
            try:
                result = detector.predict(user_text, model_choice=model_choice, use_keyword_rules=use_rules)
                st.session_state.last_result = result
                st.session_state.last_text = user_text
                st.session_state.last_model = model_choice
                st.session_state.last_field = field
                
                # Force generation to run immediately upon a fresh submission click
                trigger_generation = True 
                st.session_state.current_guidance = None 
            except RuntimeError as e:
                st.error(str(e))
    elif submit:
        st.warning("Please describe your study challenge before analyzing.")

    result = st.session_state.last_result
    if result:
        st.divider()
        st.subheader("Emotion Breakdown")

        if st.session_state.last_model == "both" and "bilstm" in result and "bert" in result:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**BiLSTM**")
                fig = px.bar(x=list(result["bilstm"].keys()), y=list(result["bilstm"].values()),
                             labels={"x": "Emotion", "y": "Probability"}, range_y=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown("**BERT**")
                fig = px.bar(x=list(result["bert"].keys()), y=list(result["bert"].values()),
                             labels={"x": "Emotion", "y": "Probability"}, range_y=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.bar(x=list(result["final"].keys()), y=list(result["final"].values()),
                         labels={"x": "Emotion", "y": "Probability"}, range_y=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**Top emotion:** `{result['top_emotion']}`")
        st.markdown(f"**Mixed emotions detected:** {', '.join(result['mixed_emotions'])}")

        if show_ai:
            st.divider()
            st.subheader("Personalized Guidance")
            
            if st.button("🔄 Regenerate response"):
                trigger_generation = True
                regenerate = True

            # Call the external API ONLY if requested, otherwise display the cached text block
            if trigger_generation or st.session_state.current_guidance is None:
                with st.spinner("Generating personalized guidance..."):
                    guidance = generate_guidance(
                        st.session_state.last_text, result["top_emotion"],
                        result["mixed_emotions"], field=st.session_state.last_field,
                        regenerate=regenerate,
                    )
                    st.session_state.current_guidance = guidance
                    
                    # Log data securely only upon fresh evaluation cycles
                    log_interaction(
                        text=st.session_state.last_text,
                        model_used=st.session_state.last_model,
                        top_emotion=result["top_emotion"],
                        mixed_emotions=result["mixed_emotions"],
                        confidence=result["final"][result["top_emotion"]],
                        guidance=guidance,
                    )
            
            # Render the structural text payload directly out of the state container array
            st.info(st.session_state.current_guidance)
            st.caption("Saved to interaction log ✅")

# =========================================================
# TAB 2: ANALYTICS DASHBOARD
# =========================================================
with tab_dashboard:
    header_col, clear_col = st.columns([4, 1])
    with header_col:
        st.subheader("Interaction History & Emotion Trends")
    with clear_col:
        if st.button("🗑️ Clear History"):
            clear_log()
            st.success("Interaction history cleared.")
            st.rerun()

    log_df = load_interaction_log()

    if log_df.empty:
        st.info("No interactions logged yet. Use the Assistant tab first.")
    else:
        log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])

        c1, c2 = st.columns(2)
        with c1:
            counts = log_df["top_emotion"].value_counts().reindex(EMOTION_LABELS, fill_value=0)
            fig = px.pie(names=counts.index, values=counts.values, title="Emotion Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            trend = log_df.groupby([log_df["timestamp"].dt.date, "top_emotion"]).size().reset_index(name="count")
            fig = px.line(trend, x="timestamp", y="count", color="top_emotion", title="Emotion Trend Over Time")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Model usage**")
        st.bar_chart(log_df["model_used"].value_counts())

        st.markdown("**Recent interactions**")
        st.dataframe(log_df.sort_values("timestamp", ascending=False).head(20), use_container_width=True)