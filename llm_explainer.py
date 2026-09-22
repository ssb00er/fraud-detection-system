"""
Fraud Detection Project: LLM Explanation Layer (Gemini version)
Takes structured SHAP feature contributions and generates a plain-English
explanation for a fraud analyst using Google's Gemini API (free tier via
Google AI Studio - no credit card required).

Setup:
    1. Go to https://aistudio.google.com/apikey and create a free API key.
    2. Set it as an environment variable: GEMINI_API_KEY
    3. pip install google-generativeai

Usage (standalone test):
    python llm_explainer.py
"""


from google import genai
from google.genai import types
from config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-3.6-flash"

SYSTEM_PROMPT = """You are assisting a fraud analyst at a bank. You will be given:
- A model's fraud probability score for a transaction
- The transaction's decision threshold
- A list of the top features that contributed to the model's decision, along with
  their SHAP values (how much each pushed the prediction toward or away from "fraud")
  and their actual values for this transaction.

Write a short, plain-English explanation (3-4 sentences max) of why the model flagged
(or did not flag) this transaction. Rules:
- Do NOT use jargon like "SHAP value" or "feature importance" in your explanation -
  translate it into plain business language (e.g., "unusually high transaction amount"
  or "an unusual pattern compared to typical transactions").
- Note that some features (V1-V28) are anonymized and cannot be tied to a specific
  real-world meaning - for those, describe them generically as "an unusual pattern in
  the transaction's underlying characteristics" rather than inventing a false meaning.
- Be direct and factual. Do not speculate beyond what the data shows.
- End with a one-sentence recommended action for the analyst (e.g., "recommend manual review"
  or "no action needed").
"""



def build_user_prompt(fraud_probability, threshold, is_flagged, top_features):
    feature_lines = "\n".join(
        f"- {f['feature']}: SHAP value = {f['shap_value']:.3f}, actual value = {f['feature_value']:.3f}"
        for f in top_features
    )
    return f"""Transaction fraud probability: {fraud_probability:.4f}
Decision threshold: {threshold}
Flagged as fraud: {is_flagged}

Top contributing features:
{feature_lines}

Explain this result for a fraud analyst."""


def generate_explanation(fraud_probability, threshold, is_flagged, top_features):
    """
    top_features: list of dicts like
        [{"feature": "v14", "shap_value": 5.18, "feature_value": -12.62}, ...]
    Returns: plain-English explanation string.
    """
    user_prompt = build_user_prompt(fraud_probability, threshold, is_flagged, top_features)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=800,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text


if __name__ == "__main__":
    # Standalone test using the example fraud case from Step 5's SHAP output
    example_features = [
        {"feature": "v14", "shap_value": 5.183257, "feature_value": -12.623316},
        {"feature": "v4", "shap_value": 1.731180, "feature_value": 7.963928},
        {"feature": "v10", "shap_value": 1.599227, "feature_value": -11.589748},
        {"feature": "v12", "shap_value": 1.585829, "feature_value": -13.542096},
        {"feature": "v17", "shap_value": 0.936556, "feature_value": -23.241597},
    ]

    explanation = generate_explanation(
        fraud_probability=1.0000,
        threshold=0.9866,
        is_flagged=True,
        top_features=example_features,
    )
    print("Generated explanation:\n")
    print(explanation)