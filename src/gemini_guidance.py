"""
Epic 4: AI-Powered Guidance & Regeneration Engine.

Given the student's raw text + detected emotion(s), ask Gemini for a short,
empathetic, field-specific response: encouragement + concrete next steps + tips.
Supports "regenerate" (re-roll with slightly higher temperature/variation).
"""
import os
import sys
import textwrap
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GEMINI_API_KEY, GEMINI_MODEL_NAME, FALLBACK_TEMPLATES_PATH

try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = bool(GEMINI_API_KEY)
except ImportError:
    GEMINI_AVAILABLE = False


PROMPT_TEMPLATE = textwrap.dedent("""
You are an empathetic academic support assistant helping a student in the
field of {field}.

Student's message: "{text}"
Detected primary emotion: {primary_emotion}
Other emotions present: {mixed_emotions}

Write a short, warm, and encouraging response with exactly three parts:
1. Encouragement (1-2 sentences acknowledging how they feel, without being generic)
2. Next Steps (2-3 concrete, actionable steps relevant to their specific problem and field)
3. Tip (one practical study/debugging tip relevant to {field})

Keep the whole response under 120 words. Do not repeat the student's message verbatim.
Format as:
Encouragement: ...
Next Steps: ...
Tip: ...
""").strip()


def _load_fallback_templates() -> pd.DataFrame:
    """Loads data/emotion_response_examples.csv (emotion, field, encouragement,
    next_steps, tip). Falls back to an empty frame if the file is missing,
    in which case get_fallback() below returns a generic hardcoded message."""
    if os.path.exists(FALLBACK_TEMPLATES_PATH):
        return pd.read_csv(FALLBACK_TEMPLATES_PATH)
    return pd.DataFrame(columns=["emotion", "field", "encouragement", "next_steps", "tip"])


_FALLBACK_DF = _load_fallback_templates()


def get_fallback(primary_emotion: str, field: str = "General") -> str:
    """Field-specific row if one exists, else the General row for that
    emotion, else a generic message."""
    df = _FALLBACK_DF
    row = df[(df["emotion"] == primary_emotion) & (df["field"] == field)]
    if row.empty:
        row = df[(df["emotion"] == primary_emotion) & (df["field"] == "General")]
    if row.empty:
        return ("Encouragement: Whatever you're feeling about this right now is a valid part of learning.\n"
                 "Next Steps: 1) Break the problem into the smallest piece you can restate clearly. 2) Work through one concrete example.\n"
                 "Tip: Explaining the sticking point out loud often reveals exactly where the gap is.")
    r = row.iloc[0]
    return f"Encouragement: {r['encouragement']}\nNext Steps: {r['next_steps']}\nTip: {r['tip']}"


def _call_gemini(prompt: str, temperature: float) -> str:
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(temperature=temperature, max_output_tokens=4096),
    )
    # Diagnostics: show exactly where the token budget went, and whether the
    # response was cut short, so we fix the real cause instead of guessing.
    try:
        finish_reason = response.candidates[0].finish_reason
        usage = response.usage_metadata
        print(f"[gemini_guidance] finish_reason={finish_reason} | "
              f"prompt_tokens={usage.prompt_token_count} | "
              f"output_tokens={usage.candidates_token_count} | "
              f"total_tokens={usage.total_token_count}")
        if int(finish_reason) not in (1,):  # 1 = STOP (normal completion)
            print(f"[gemini_guidance] Warning: response may be truncated (finish_reason={finish_reason})")
    except Exception as diag_err:
        print(f"[gemini_guidance] (diagnostics unavailable: {diag_err})")
    return response.text.strip()


def generate_guidance(text: str, primary_emotion: str, mixed_emotions: list,
                       field: str = "General", regenerate: bool = False) -> str:
    """
    Returns a formatted guidance string. Falls back to a field-aware CSV
    template if Gemini isn't configured (e.g. no API key set), so the app
    still works end-to-end.
    """
    if not GEMINI_AVAILABLE:
        return get_fallback(primary_emotion, field) + \
            "\n\n(Set GEMINI_API_KEY as an environment variable to enable live AI-generated guidance.)"

    prompt = PROMPT_TEMPLATE.format(
        text=text,
        primary_emotion=primary_emotion,
        mixed_emotions=", ".join(mixed_emotions) if mixed_emotions else "None",
        field=field,
    )
    temperature = 0.9 if regenerate else 0.6
    try:
        return _call_gemini(prompt, temperature)
    except Exception as e:
        return f"(Gemini call failed: {e})\n\n" + get_fallback(primary_emotion, field)


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(generate_guidance("I'm so lost on recursion", "Confused", ["Confused", "Curious"], field="Computer Science / Programming"))
