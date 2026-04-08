#!/usr/bin/env python3
"""
OpenEnv compatible inference (NO external API dependency)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from credit_card_env import CreditCardEnv, Action, Observation

logging.basicConfig(level=logging.WARNING)

TASK_CONFIGS: List[Dict[str, Any]] = [
    {"task_name": "easy", "display_name": "card_recommendation_easy"},
    {"task_name": "medium", "display_name": "transaction_optimization_medium"},
    {"task_name": "hard", "display_name": "portfolio_optimization_hard"},
]


# -------------------------------------------------------------------
# HEURISTIC AGENT (IMPORTANT - NO API)
# -------------------------------------------------------------------
def smart_agent(obs: Observation) -> Action:
    user = obs.user_profile
    target_category = user.top_spending_category

    if obs.current_transaction:
        target_category = obs.current_transaction.category

    candidates = obs.available_cards
    owned = {c.lower().strip() for c in user.owned_cards}

    # Filtering logic
    if "medium" in obs.task_name:
        candidates = [c for c in candidates if c.card_name.lower().strip() in owned]
    else:
        candidates = [c for c in candidates if c.card_name.lower().strip() not in owned]

    if not candidates:
        candidates = obs.available_cards

    def score(c):
        s = 0
        if str(c.primary_category) == str(target_category):
            s += 100
        elif target_category in [str(cat) for cat in c.categories]:
            s += 50
        s += c.reward_rate
        s -= (c.annual_fee / 1000)
        return s

    best = max(candidates, key=score)

    return Action(
        recommended_card=best.card_name,
        reasoning=f"Best for {target_category}"
    )


# -------------------------------------------------------------------
# RUN TASK
# -------------------------------------------------------------------
def run_task(env: CreditCardEnv, cfg: Dict[str, Any]) -> float:
    print(f"\n[START] {cfg['display_name']}")

    obs = env.reset(task_name=cfg["task_name"])
    done = False

    while not done:
        action = smart_agent(obs)
        obs, reward, done, _ = env.step(action)
        print(f"[STEP] {action.recommended_card} → {reward.score:.3f}")

    score = env.grade()
    print(f"[END] Score = {score:.3f}")
    return score


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    env = CreditCardEnv(seed=42)

    results = {}
    for cfg in TASK_CONFIGS:
        results[cfg["display_name"]] = run_task(env, cfg)

    print("\nFINAL RESULTS")
    avg = sum(results.values()) / len(results)

    for k, v in results.items():
        print(f"{k}: {v:.3f}")

    print(f"AVERAGE: {avg:.3f}")


if __name__ == "__main__":
    main()
