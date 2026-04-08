"""
Reward computation module.
All reward functions accept an Action and contextual state, returning a Reward object.
Scores are always clamped to [0.0, 1.0].
"""
from __future__ import annotations

import logging
from typing import List, Optional

from credit_card_env.models import (
    Action,
    CreditCard,
    Reward,
    RewardBreakdown,
    SpendingCategory,
    Transaction,
    UserProfile,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Score weights
# ---------------------------------------------------------------------------
W_CATEGORY_MATCH = 0.30
W_REWARD_RATE = 0.30
W_FEE_EFFICIENCY = 0.15
W_NOVELTY = 0.15
W_WELCOME = 0.10

REDUNDANCY_PENALTY = 0.25
ELIGIBILITY_PENALTY = 0.40


def _find_card(card_name: str, catalogue: List[CreditCard]) -> Optional[CreditCard]:
    """Case-insensitive card lookup."""
    name_lower = card_name.strip().lower()
    for card in catalogue:
        if card.card_name.strip().lower() == name_lower:
            return card
    return None


def _normalize_reward_rate(rate: float, max_rate: float = 10.0) -> float:
    """Normalize reward rate to [0, 1]."""
    return min(rate / max_rate, 1.0)


def _credit_tier_rank(tier: str) -> int:
    ranks = {"poor": 0, "fair": 1, "good": 2, "very_good": 3, "excellent": 4}
    return ranks.get(tier, 0)


# ---------------------------------------------------------------------------
# Core reward function
# ---------------------------------------------------------------------------

def compute_recommendation_reward(
    action: Action,
    user: UserProfile,
    catalogue: List[CreditCard],
    target_category: Optional[SpendingCategory] = None,
    require_new_card: bool = True,
) -> Reward:
    """
    Compute a reward score for a card recommendation action.

    Args:
        action: The agent's recommended card action.
        user: The user's current profile.
        catalogue: Full list of available credit cards.
        target_category: If set, reward is focused on this spending category.
        require_new_card: If True, penalise recommending an already-owned card.

    Returns:
        Reward with score in [0.0, 1.0] and detailed breakdown.
    """
    card = _find_card(action.recommended_card, catalogue)

    if card is None:
        return Reward(
            score=0.0,
            breakdown=RewardBreakdown(),
            rationale=f"Card '{action.recommended_card}' not found in catalogue. Score: 0.0",
            action_taken=action.recommended_card,
        )

    breakdown = RewardBreakdown()
    rationale_parts: list[str] = []

    # ------------------------------------------------------------------
    # 1. Eligibility check (hard penalty)
    # ------------------------------------------------------------------
    user_tier_rank = _credit_tier_rank(user.credit_score_tier)
    required_rank = _credit_tier_rank(card.credit_score_required)

    if user_tier_rank < required_rank:
        breakdown.eligibility_penalty = -ELIGIBILITY_PENALTY
        rationale_parts.append(
            f"User credit tier '{user.credit_score_tier}' below required '{card.credit_score_required}' "
            f"(penalty {ELIGIBILITY_PENALTY})"
        )

    if user.annual_income < card.min_income_required:
        breakdown.eligibility_penalty = min(breakdown.eligibility_penalty - 0.15, -0.55)
        rationale_parts.append(
            f"User income {user.annual_income} below required {card.min_income_required}"
        )

    # ------------------------------------------------------------------
    # 2. Redundancy check
    # ------------------------------------------------------------------
    is_owned = card.card_name in user.owned_cards
    if is_owned and require_new_card:
        breakdown.redundancy_penalty = -REDUNDANCY_PENALTY
        rationale_parts.append(
            f"User already owns '{card.card_name}' (redundancy penalty {REDUNDANCY_PENALTY})"
        )

    # ------------------------------------------------------------------
    # 3. Category match
    # ------------------------------------------------------------------
    effective_category = target_category or user.top_spending_category
    
    # Get the string value of categories for comparison
    target_cat_val = effective_category.value if hasattr(effective_category, "value") else effective_category
    primary_cat_val = card.primary_category.value if hasattr(card.primary_category, "value") else card.primary_category

    if primary_cat_val == target_cat_val:
        breakdown.category_match = W_CATEGORY_MATCH
        rationale_parts.append(
            f"Primary category '{primary_cat_val}' perfectly matches target '{target_cat_val}' (+{W_CATEGORY_MATCH})"
        )
    elif target_cat_val in [c.value if hasattr(c, "value") else c for c in card.categories]:
        breakdown.category_match = W_CATEGORY_MATCH * 0.6
        rationale_parts.append(
            f"Target category '{target_cat_val}' supported (not primary) (+{W_CATEGORY_MATCH * 0.6:.2f})"
        )
    else:
        breakdown.category_match = 0.0
        rationale_parts.append(f"Category '{effective_category}' not in card's supported list (+0.0)")

    # ------------------------------------------------------------------
    # 4. Reward rate score
    # ------------------------------------------------------------------
    effective_rate = card.category_reward_rate(effective_category)
    breakdown.reward_rate = W_REWARD_RATE * _normalize_reward_rate(effective_rate)
    rationale_parts.append(
        f"Reward rate {effective_rate:.2f}% for '{effective_category}' → score +{breakdown.reward_rate:.3f}"
    )

    # ------------------------------------------------------------------
    # 5. Fee efficiency
    # ------------------------------------------------------------------
    if card.annual_fee == 0:
        breakdown.fee_efficiency = W_FEE_EFFICIENCY
        rationale_parts.append(f"No annual fee — max efficiency (+{W_FEE_EFFICIENCY})")
    elif card.net_annual_value > 0:
        ratio = min(card.net_annual_value / max(card.annual_fee, 1), 5.0) / 5.0
        breakdown.fee_efficiency = W_FEE_EFFICIENCY * ratio
        rationale_parts.append(
            f"Annual benefit {card.annual_benefit_value} vs fee {card.annual_fee} "
            f"→ +{breakdown.fee_efficiency:.3f}"
        )
    else:
        breakdown.fee_efficiency = 0.0
        rationale_parts.append("Annual fee exceeds estimated benefit (+0.0)")

    # ------------------------------------------------------------------
    # 6. Novelty / welcome offer bonus
    # ------------------------------------------------------------------
    if not is_owned:
        breakdown.novelty_bonus = W_NOVELTY
        rationale_parts.append(f"New card for user (+{W_NOVELTY})")
    else:
        breakdown.novelty_bonus = 0.0

    if card.welcome_offer and not is_owned:
        breakdown.welcome_offer_bonus = W_WELCOME
        rationale_parts.append(f"Welcome offer available: '{card.welcome_offer[:60]}' (+{W_WELCOME})")

    # ------------------------------------------------------------------
    # Final score
    # ------------------------------------------------------------------
    raw = (
        breakdown.category_match
        + breakdown.reward_rate
        + breakdown.fee_efficiency
        + breakdown.novelty_bonus
        + breakdown.welcome_offer_bonus
        + breakdown.eligibility_penalty
        + breakdown.redundancy_penalty
    )
    final_score = max(0.0, min(1.0, raw))

    rationale = "; ".join(rationale_parts) + f" | TOTAL: {final_score:.3f}"
    logger.debug("Reward for '%s': %.3f", action.recommended_card, final_score)

    return Reward(
        score=final_score,
        breakdown=breakdown,
        rationale=rationale,
        action_taken=action.recommended_card,
    )


# ---------------------------------------------------------------------------
# Transaction-specific reward function
# ---------------------------------------------------------------------------

def compute_transaction_reward(
    action: Action,
    user: UserProfile,
    transaction: Transaction,
    catalogue: List[CreditCard],
) -> Reward:
    """
    Compute reward for selecting a card to pay for a transaction.
    The recommended card MUST be one the user already owns.
    Only cards in user.owned_cards are valid candidates.
    """
    card = _find_card(action.recommended_card, catalogue)

    if card is None:
        return Reward(
            score=0.0,
            breakdown=RewardBreakdown(),
            rationale=f"Card '{action.recommended_card}' not found.",
            action_taken=action.recommended_card,
        )

    # Owership check: for MEDIUM tasks, recommended card must be owned
    if card.card_name not in user.owned_cards:
        return Reward(
            score=0.0,
            breakdown=RewardBreakdown(eligibility_penalty=-1.0),
            rationale=f"User doesn't own '{card.card_name}'. Must recommend an owned card.",
            action_taken=action.recommended_card,
        )

    breakdown = RewardBreakdown()
    rationale_parts: list[str] = []

    # Find best possible rate among all owned cards for this transaction category
    target_cat_val = transaction.category.value if hasattr(transaction.category, "value") else transaction.category
    
    # Simple converter to ensure we always have the Enum for card.category_reward_rate if it expects it
    from credit_card_env.models import SpendingCategory
    try:
        sc_category = SpendingCategory(target_cat_val)
    except:
        sc_category = SpendingCategory.GENERAL

    owned_cards_objs = [c for c in catalogue if c.card_name in user.owned_cards]
    best_rate = max(
        (c.category_reward_rate(sc_category) for c in owned_cards_objs),
        default=1.0,
    )

    effective_rate = card.category_reward_rate(sc_category)

    # Category match score
    primary_cat_val = card.primary_category.value if hasattr(card.primary_category, "value") else card.primary_category
    if primary_cat_val == target_cat_val:
        breakdown.category_match = W_CATEGORY_MATCH
        rationale_parts.append(f"Primary category match (+{W_CATEGORY_MATCH})")
    elif target_cat_val in [c.value if hasattr(c, "value") else c for c in card.categories]:
        breakdown.category_match = W_CATEGORY_MATCH * 0.6
        rationale_parts.append(f"Category supported (+{W_CATEGORY_MATCH * 0.6:.2f})")

    # Reward rate: score relative to best available rate
    rate_ratio = effective_rate / max(best_rate, 0.01)
    breakdown.reward_rate = W_REWARD_RATE * 2 * min(rate_ratio, 1.0)  # doubled weight for transaction task
    rationale_parts.append(
        f"Reward rate {effective_rate:.2f}% (best: {best_rate:.2f}%) → rate_ratio={rate_ratio:.2f} "
        f"→ +{breakdown.reward_rate:.3f}"
    )

    # Cashback earned on transaction amount
    cashback_earned = (effective_rate / 100) * transaction.amount
    rationale_parts.append(f"Estimated cashback on txn: INR {cashback_earned:.2f}")

    # Fee efficiency
    breakdown.fee_efficiency = W_FEE_EFFICIENCY if card.annual_fee == 0 else W_FEE_EFFICIENCY * 0.5

    raw = breakdown.category_match + breakdown.reward_rate + breakdown.fee_efficiency
    final_score = max(0.0, min(1.0, raw))

    return Reward(
        score=final_score,
        breakdown=breakdown,
        rationale="; ".join(rationale_parts) + f" | TOTAL: {final_score:.3f}",
        action_taken=action.recommended_card,
    )
