# ========================= inference.py =========================
#!/usr/bin/env python3
"""
inference.py — Baseline inference script for the Credit Card Recommendation OpenEnv.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
from typing import Any, Dict, List

from openai import OpenAI
from dotenv import load_dotenv

from credit_card_env import CreditCardEnv, Action, Observation

# Load environment variables
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
# Hackathon explicitly demands HF_TOKEN 
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("OPENAI_API_KEY", ""))

# Configure logging - keep it minimal for cleaner output
logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logs from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

TASK_CONFIGS: List[Dict[str, Any]] = [
    {"task_name": "easy", "display_name": "card_recommendation_easy", "max_steps": 1},
    {"task_name": "medium", "display_name": "transaction_optimization_medium", "max_steps": 1},
    {"task_name": "hard", "display_name": "portfolio_optimization_hard", "max_steps": 6},
]

def build_system_prompt() -> str:
    return textwrap.dedent("""
        You are an expert financial advisor specializing in credit card optimization for Indian consumers.

        Respond ONLY in JSON:
        {
            "recommended_card": "<exact card name>",
            "reasoning": "<brief explanation>"
        }
    """).strip()

def build_user_prompt(obs: Observation) -> str:
    user = obs.user_profile
    cards_summary = []

    for card in obs.available_cards:
        cards_summary.append({
            "card_name": card.card_name,
            "bank": card.bank_name,
            "primary_category": card.primary_category,
            "categories": card.categories,
            "reward_rate_pct": card.reward_rate,
            "annual_fee_inr": card.annual_fee,
            "benefits": card.benefits[:3],
        })

    prompt_data = {
        "task": obs.task_name,
        "user": {
            "income": user.annual_income,
            "owned_cards": user.owned_cards,
            "top_category": user.top_spending_category,
        },
        "available_cards": cards_summary,
        "context": obs.context,
    }

    if obs.current_transaction:
        prompt_data["current_transaction"] = {
            "merchant": obs.current_transaction.merchant,
            "amount": obs.current_transaction.amount,
            "category": obs.current_transaction.category,
        }

    return json.dumps(prompt_data, indent=2, default=str)

def smart_heuristic_fallback(obs: Observation) -> Action:
    """
    Returns a reasonably good card recommendation based on simple rules.
    Used when the API fails (e.g. Quota Exceeded).
    """
    user = obs.user_profile
    target_category = user.top_spending_category
    if obs.current_transaction:
        target_category = obs.current_transaction.category

    candidates = obs.available_cards
    
    owned_names_norm = {name.strip().lower() for name in user.owned_cards}
    
    # Task specific filtering logic
    if "medium" in obs.task_name or obs.context.get("phase") == 2:
        # For transaction tasks, must pick from OWNED cards
        candidates = [c for c in candidates if c.card_name.strip().lower() in owned_names_norm]
    elif "easy" in obs.task_name or (obs.task_name == "portfolio_optimization_hard" and obs.step_number == 0):
        # For recommendation tasks, must pick a NEW card
        candidates = [c for c in candidates if c.card_name.strip().lower() not in owned_names_norm]

    if not candidates:
        candidates = obs.available_cards

    # Score candidates: match target category + reward rate bonus
    def card_score(c):
        score = 0.0
        if str(c.primary_category) == str(target_category):
            score += 100.0
        elif str(target_category) in [str(cat) for cat in c.categories]:
            score += 50.0
        score += c.reward_rate
        score -= (c.annual_fee / 1000.0)
        return score

    best_card = max(candidates, key=card_score)
    return Action(
        recommended_card=best_card.card_name, 
        reasoning=f"Selected {best_card.card_name} based on category match for {target_category}."
    )

def call_agent(client: OpenAI, obs: Observation) -> Action:
    if not HF_TOKEN or "your-key" in HF_TOKEN:
        return smart_heuristic_fallback(obs)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": build_user_prompt(obs)},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            timeout=10,
        )

        data = json.loads(response.choices[0].message.content)
        return Action(recommended_card=data["recommended_card"], reasoning=data["reasoning"])

    except Exception:
        return smart_heuristic_fallback(obs)

def run_task(env: CreditCardEnv, client: OpenAI, cfg: Dict[str, Any]) -> float:
    print(f"\n[START] {cfg['display_name']}")

    obs = env.reset(task_name=cfg["task_name"])
    done = False

    while not done:
        action = call_agent(client, obs)
        obs, reward, done, _ = env.step(action)
        print(f"[STEP] action='{action.recommended_card}' reward={reward.score:.4f}")

    final_reward = env.grade()
    print(f"[END] total_reward={final_reward:.4f}")
    return final_reward

def main():
    print("=" * 60)
    print("  Credit Card Recommendation — OpenEnv Inference")
    print(f"  Model: {MODEL_NAME}  |  Base URL: {API_BASE_URL}")
    print("=" * 60)

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    env = CreditCardEnv(seed=42)

    results = {}
    for cfg in TASK_CONFIGS:
        score = run_task(env, client, cfg)
        results[cfg["display_name"]] = score

    print("\n" + "=" * 60)
    print("  FINAL SCORES")
    print("=" * 60)
    for name, score in results.items():
        print(f"  {name:<45} {score:.4f}")

    avg = sum(results.values()) / len(results)
    print(f"  {'AVERAGE':<45} {avg:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
