import gradio as gr
import os
from openai import OpenAI
from credit_card_env import CreditCardEnv
from ask import get_unique_cards, build_compact_database, ask_ai_expert

# Setup
env = CreditCardEnv(seed=42)

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)

# Prepare DB once (IMPORTANT)
unique_catalogue = get_unique_cards(env.catalogue)
compact_db = build_compact_database(unique_catalogue)


def recommend(query):
    if not OPENAI_API_KEY:
        return "❌ API key missing. Please set OPENAI_API_KEY in Hugging Face Secrets."

    result = ask_ai_expert(query, compact_db, client)

    if not result or "top_cards" not in result:
        return "⚠️ Failed to generate recommendation."

    output = f"🧠 AI Insight:\n{result.get('analysis','')}\n\n"

    for i, card in enumerate(result["top_cards"], 1):
        output += f"""
{i}. {card.get('card_name')} ({card.get('bank_name')})
   Fee: ₹{card.get('annual_fee')}
   Reason: {card.get('ai_reasoning')}

"""

    return output


iface = gr.Interface(
    fn=recommend,
    inputs=gr.Textbox(
        placeholder="e.g. I want a lifetime free card for food delivery",
        label="Ask AI Advisor"
    ),
    outputs="text",
    title="💳 AI Credit Card Advisor",
    description="Ask anything and get top 3 credit card recommendations"
)

iface.launch(server_name="0.0.0.0", server_port=7860)