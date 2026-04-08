"""
HARD Task: Portfolio Optimization — Multi-Step Scenario.

Objective:
    Phase 1 (Step 1): Recommend a new credit card to add to the user's portfolio.
    Phase 2 (Steps 2–6): Route 5 sequential transactions to the best card
                         (including the newly recommended card if adopted).

Input State:
    - UserProfile with existing cards and diverse spending habits
    - 5 pre-defined transactions across different categories
    - Full credit card catalogue

Expected Behaviour:
    - Phase 1: Recommend a strategically complementary card
    - Phase 2: Use the optimal card for each micro-transaction
    - Agent must consider the newly added card as part of the portfolio

Grader:
    - Phase 1 contributes 30% of the total score
    - Phase 2 contributes 70% (14% per transaction, 5 transactions)
    - Final score: weighted average in [0.0, 1.0]
"""
from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional

from credit_card_env.models import (
    Action,
    CreditCard,
    CreditScoreTier,
    Observation,
    Reward,
    RewardBreakdown,
    SpendingCategory,
    TaskDifficulty,
    Transaction,
    UserProfile,
)
from credit_card_env.reward import compute_recommendation_reward, compute_transaction_reward
from credit_card_env.tasks.base import BaseTask

# ---------------------------------------------------------------------------
# Default hard scenario
# ---------------------------------------------------------------------------
_HARD_USER = UserProfile(
    user_id="u_hard_01",
    name="Rohan Verma",
    annual_income=1_200_000.0,
    credit_score_tier=CreditScoreTier.VERY_GOOD,
    owned_cards=["HDFC MoneyBack+", "ICICI Coral"],
    monthly_spending={
        SpendingCategory.TRAVEL: 20000,
        SpendingCategory.DINING: 12000,
        SpendingCategory.FUEL: 5000,
        SpendingCategory.ONLINE: 8000,
        SpendingCategory.ENTERTAINMENT: 4000,
    },
)

_HARD_TRANSACTIONS = [
    Transaction(merchant="IndiGo Airlines", amount=12000.0, category=SpendingCategory.TRAVEL),
    Transaction(merchant="Zomato", amount=2500.0, category=SpendingCategory.DINING),
    Transaction(merchant="BPCL Pump", amount=4000.0, category=SpendingCategory.FUEL),
    Transaction(merchant="Amazon", amount=3500.0, category=SpendingCategory.ONLINE),
    Transaction(merchant="PVR Cinemas", amount=1200.0, category=SpendingCategory.ENTERTAINMENT),
]

# Weight distribution: 30% for portfolio card, 70% for 5 transactions (14% each)
_PHASE1_WEIGHT = 0.30
_PHASE2_WEIGHT = 0.70
_TXN_WEIGHT = _PHASE2_WEIGHT / len(_HARD_TRANSACTIONS)  # 0.14 each


class PortfolioOptimizationTask(BaseTask):
    """
    HARD — Multi-step portfolio optimization:
    Step 1 → recommend a new card to add to portfolio.
    Steps 2–6 → route 5 transactions to optimal owned card.
    """

    name = "portfolio_optimization_hard"
    difficulty = TaskDifficulty.HARD
    description = (
        "Phase 1: Recommend a new credit card to strategically expand the user's portfolio. "
        "Phase 2: Route 5 consecutive transactions to the best available card, "
        "including the newly added card if the agent chooses to adopt it."
    )
    max_steps = 6  # 1 portfolio step + 5 transaction steps

    def __init__(self, catalogue: List[CreditCard], seed: Optional[int] = None):
        super().__init__(catalogue)
        self._rng = random.Random(seed)
        self._current_user: Optional[UserProfile] = None
        self._transactions: List[Transaction] = []
        self._step_count = 0
        self._action_history: List[Action] = []
        self._obs_history: List[Observation] = []
        self._phase1_reward: Optional[Reward] = None
        self._phase2_rewards: List[Reward] = []
        self._newly_added_card: Optional[str] = None

    def reset(self, user: Optional[UserProfile] = None) -> Observation:
        self._step_count = 0
        self._action_history = []
        self._obs_history = []
        self._phase1_reward = None
        self._phase2_rewards = []
        self._newly_added_card = None

        self._current_user = copy.deepcopy(user or _HARD_USER)
        self._transactions = copy.deepcopy(_HARD_TRANSACTIONS)

        obs = Observation(
            user_profile=self._current_user,
            available_cards=self.catalogue,
            current_transaction=None,
            step_number=0,
            max_steps=self.max_steps,
            task_name=self.name,
            task_difficulty=self.difficulty,
            context={
                "phase": 1,
                "objective": self.description,
                "instruction": (
                    f"PHASE 1: {self._current_user.name} already owns: {self._current_user.owned_cards}. "
                    f"Recommend ONE new credit card to add to their portfolio to complement their spending habits."
                    f" After this, you will be asked to route 5 transactions to the best card."
                ),
                "upcoming_transactions": [t.dict() for t in self._transactions],
            },
        )
        self._obs_history.append(obs)
        return obs

    def step(
        self, action: Action, observation: Observation
    ) -> tuple[Observation, Reward, bool, Dict[str, Any]]:
        self._step_count += 1
        self._action_history.append(action)
        done = self._step_count >= self.max_steps

        # ------------------------------------------------------------------
        # PHASE 1: Portfolio card recommendation
        # ------------------------------------------------------------------
        if self._step_count == 1:
            reward = compute_recommendation_reward(
                action=action,
                user=self._current_user,
                catalogue=self.catalogue,
                target_category=self._current_user.top_spending_category,
                require_new_card=True,
            )
            self._phase1_reward = reward

            # Tentatively add the card to user's portfolio for phase 2
            from credit_card_env.reward import _find_card
            rec_card = _find_card(action.recommended_card, self.catalogue)
            if rec_card and action.recommended_card not in self._current_user.owned_cards:
                self._current_user = self._current_user.copy(
                    update={"owned_cards": self._current_user.owned_cards + [action.recommended_card]}
                )
                self._newly_added_card = action.recommended_card

            next_obs = Observation(
                user_profile=self._current_user,
                available_cards=self.catalogue,
                current_transaction=self._transactions[0] if self._transactions else None,
                step_number=self._step_count,
                max_steps=self.max_steps,
                task_name=self.name,
                task_difficulty=self.difficulty,
                context={
                    "phase": 2,
                    "instruction": (
                        f"PHASE 2 — Transaction 1 of 5: "
                        f"Updated portfolio: {self._current_user.owned_cards}. "
                        f"Recommend the BEST card from your portfolio for this transaction."
                    ),
                    "remaining_transactions": len(self._transactions),
                    "newly_added_card": self._newly_added_card,
                },
            )

        # ------------------------------------------------------------------
        # PHASE 2: Transaction routing (steps 2–6)
        # ------------------------------------------------------------------
        else:
            txn_index = self._step_count - 2  # 0-indexed
            current_txn = self._transactions[txn_index]

            reward = compute_transaction_reward(
                action=action,
                user=self._current_user,
                transaction=current_txn,
                catalogue=self.catalogue,
            )
            self._phase2_rewards.append(reward)

            next_txn_index = txn_index + 1
            next_txn = self._transactions[next_txn_index] if next_txn_index < len(self._transactions) else None

            next_obs = Observation(
                user_profile=self._current_user,
                available_cards=self.catalogue,
                current_transaction=next_txn,
                step_number=self._step_count,
                max_steps=self.max_steps,
                task_name=self.name,
                task_difficulty=self.difficulty,
                context={
                    "phase": 2,
                    "instruction": (
                        f"PHASE 2 — Transaction {txn_index + 1} of 5 completed. "
                        + (
                            f"Transaction {txn_index + 2} of 5: "
                            f"Recommend the BEST card for the next transaction."
                            if next_txn else "All transactions complete."
                        )
                    ),
                    "remaining_transactions": max(0, len(self._transactions) - self._step_count + 1),
                },
            )

        self._obs_history.append(next_obs)
        return next_obs, reward, done, {"step": self._step_count, "phase": 1 if self._step_count == 1 else 2}

    def grade(self, actions: List[Action], observations: List[Observation]) -> float:
        """
        Weighted grader:
            Phase 1: 30% weight — portfolio card recommendation quality
            Phase 2: 70% weight — average quality across 5 transaction routings
        """
        if not actions or self._current_user is None:
            return 0.0

        # Re-run phase 1 grading
        phase1_score = 0.0
        if len(actions) >= 1:
            r = compute_recommendation_reward(
                action=actions[0],
                user=_HARD_USER,  # original user without the newly added card
                catalogue=self.catalogue,
                target_category=_HARD_USER.top_spending_category,
                require_new_card=True,
            )
            phase1_score = r.score

        # Re-run phase 2 grading
        phase2_scores: List[float] = []
        # Reconstruct portfolio after phase 1
        user_phase2 = copy.deepcopy(_HARD_USER)
        from credit_card_env.reward import _find_card
        if len(actions) >= 1:
            rec = _find_card(actions[0].recommended_card, self.catalogue)
            if rec and actions[0].recommended_card not in user_phase2.owned_cards:
                user_phase2 = user_phase2.copy(
                    update={"owned_cards": user_phase2.owned_cards + [actions[0].recommended_card]}
                )

        for i, txn in enumerate(_HARD_TRANSACTIONS):
            action_idx = i + 1  # skip phase 1 action
            if action_idx >= len(actions):
                phase2_scores.append(0.0)
                continue
            r = compute_transaction_reward(
                action=actions[action_idx],
                user=user_phase2,
                transaction=txn,
                catalogue=self.catalogue,
            )
            phase2_scores.append(r.score)

        phase2_avg = sum(phase2_scores) / len(_HARD_TRANSACTIONS) if phase2_scores else 0.0
        final = _PHASE1_WEIGHT * phase1_score + _PHASE2_WEIGHT * phase2_avg
        return round(max(0.0, min(1.0, final)), 4)
