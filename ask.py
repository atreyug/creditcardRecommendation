"""
ask.py — True AI Credit Card Advisor.
Uses the OpenAI API to evaluate your entire credit card database 
and answer natural language requests with deep contextual understanding.
"""
import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI
from credit_card_env import CreditCardEnv

# Load config
load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

logging.getLogger("httpx").setLevel(logging.WARNING)

def get_unique_cards(catalogue):
    """Deduplicates cards by name."""
    seen = set()
    unique = []
    for c in catalogue:
        n = c.card_name.strip().lower()
        if n not in seen:
            seen.add(n)
            unique.append(c)
    return unique

def build_compact_database(catalogue):
    """Creates a token-efficient JSON block of all cards to feed to the LLM."""
    db = []
    for c in catalogue:
        db.append({
            "name": c.card_name,
            "bank": c.bank_name,
            "category": str(c.primary_category),
            "fee_inr": c.annual_fee,
            "reward_rate": c.reward_rate,
            "benefits": c.benefits
        })
    return db

def ask_ai_expert(query: str, card_db: list, client: OpenAI) -> dict:
    """Passes the user's query and the ENTIRE card database to OpenAI for deep reasoning."""
    
    system_prompt = f"""
    You are an elite, highly intelligent credit card advisor for the Indian market.
    You have exclusive access to the deeply detailed credit card database provided below (JSON format).
    
    DATABASE:
    {json.dumps(card_db)}
    
    YOUR TASK:
    Understand the user's natural language request perfectly. Identify implicit needs (e.g. "student" = low/zero fee, "swiggy" = food delivery, "flights" = travel/lounge, etc).
    Thoroughly scan the provided DATABASE to find the absolute best 3 distinct credit cards that match the user's intent perfectly.
    DO NOT hallucinate cards. YOU MUST ONLY RECOMMEND CARDS FROM THE DATABASE EXACTLY AS NAMED.
    
    Return your response strictly in the following JSON format:
    {{
        "analysis": "A brief 2-sentence explanation of what you understood the user needs, and how you evaluated the database.",
        "top_cards": [
            {{
                "card_name": "Exact Name from DB",
                "bank_name": "Exact Bank from DB",
                "annual_fee": "Fee number",
                "ai_reasoning": "Detailed, highly specific explanation of exactly WHY this card is the perfect fit for this specific user's query based on its benefits."
            }},
            ... (exactly 3 cards)
        ]
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.2, # Low temperature for analytical accuracy
            response_format={"type": "json_object"},
            timeout=30 # Larger timeout since we are sending full DB
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"\n[Error] Failed to connect to OpenAI API: {e}")
        print("Please check your API Key and Quota limits in the .env file.")
        return None

def main():
    print("=" * 70)
    print("  🧠 TRUE AI-POWERED CREDIT CARD ADVISOR")
    print("=" * 70)
    
    env = CreditCardEnv(seed=42)
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)
    
    unique_catalogue = get_unique_cards(env.catalogue)
    print(f"Loaded {len(unique_catalogue)} unique cards from your database.")
    
    if not OPENAI_API_KEY or "your-key" in OPENAI_API_KEY:
        print("\n[Warning] No valid OpenAI API Key found in .env! This script requires the API to process data.")
        return
        
    print("Compressing database for the LLM context... Done.")
    compact_db = build_compact_database(unique_catalogue)
    
    while True:
        print("\n" + "-" * 70)
        query = input("Tell the AI what you need (e.g. 'I want a lifetime free card for food delivery'):\n[User] > ").strip()
        
        if query.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        if not query:
            continue

        print("\n[AI] Analyzing your request against all 104 credit cards via OpenAI...")
        
        # True AI recommendation
        result = ask_ai_expert(query, compact_db, client)
        
        if not result or "top_cards" not in result:
            continue
            
        print("\n" + "="*70)
        print(f"[AI INSIGHT]: {result.get('analysis', '')}")
        print("="*70)
        
        recs = result['top_cards']
        print(f"\n🌟 TOP {len(recs)} CARDS SELECTED BY AI:")
        for i, card in enumerate(recs, 1):
            print(f"\n[{i}] {card.get('card_name', 'Unknown')} ({card.get('bank_name', '')})")
            print(f"    Fee: ₹{card.get('annual_fee', '0')}")
            print(f"    AI Reasoning: {card.get('ai_reasoning', '')}")

if __name__ == "__main__":
    main()
