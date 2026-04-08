"""
MEDIUM Task: Transaction Optimization.

Objective:
    Given a specific transaction (amount + category), select the best card
    from the user's already-owned cards to maximise reward/cashback.

Input State:
    - UserProfile with a non-empty owned_cards list
    - A Transaction with amount and category
    - Full credit card catalogue (agent may only recommend owned cards)

Expected Behaviour:
    - Agent must choose from the user's owned cards
    - Should select the card with the highest reward rate for the transaction category
    - Recommending an unowned card results in score 0.0

Grader:
    Deterministic score in [0.0, 1.0] based on:
    - Whether the chosen card maximises cashback for the transaction category
    - Relative reward rate (chosen_rate / best_possible_rate)
    - Category alignment (primary vs supported)
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
    SpendingCategory,
    TaskDifficulty,
    Transaction,
    UserProfile,
)
from credit_card_env.reward import compute_transaction_reward
from credit_card_env.tasks.base import BaseTask

# ---------------------------------------------------------------------------
# Default scenarios: (user, transaction)
# ---------------------------------------------------------------------------
_DEFAULT_SCENARIOS = [
    (
        UserProfile(
            user_id="u_med_01",
            name="Arjun Nair",
            annual_income=700_000.0,
            credit_score_tier=CreditScoreTier.GOOD,
            owned_cards=["HDFC Regalia", "Axis Bank Ace", "ICICI Coral"],
            monthly_spending={
                SpendingCategory.TRAVEL: 10000,
                SpendingCategory.DINING: 6000,
                SpendingCategory.FUEL: 3000,
            },
        ),
        Transaction(merchant="Swiggy", amount=1500.0, category=SpendingCategory.DINING),
    ),
    (
        UserProfile(
            user_id="u_med_02",
            name="Divya Iyer",
            annual_income=500_000.0,
            credit_score_tier=CreditScoreTier.GOOD,
            owned_cards=["BPCL SBI Card Octane", "Standard Chartered Super Value Titanium", "IndusInd Tiger"],
            monthly_spending={
                SpendingCategory.FUEL: 8000,
                SpendingCategory.GROCERIES: 4000,
            },
        ),
        Transaction(merchant="BPCL Pump", amount=3000.0, category=SpendingCategory.FUEL),
    ),
    (
        UserProfile(
            user_id="u_med_03",
            name="Kabir Desai",
            annual_income=800_000.0,
            credit_score_tier=CreditScoreTier.VERY_GOOD,
            owned_cards=["SBI SimplyCLICK", "HDFC Millennia", "Kotak Mahindra PVR Platinum"],
            monthly_spending={
                SpendingCategory.ONLINE: 12000,
                SpendingCategory.ENTERTAINMENT: 5000,
            },
        ),
        Transaction(merchant="Amazon", amount=5000.0, category=SpendingCategory.ONLINE),
    ),
]


class TransactionOptimizationTask(BaseTask):
    """
    MEDIUM — Single-step: select the best owned card for a given transaction.
    """

    name = "transaction_optimization_medium"
    difficulty = TaskDifficulty.MEDIUM
    description = (
        "Given a specific purchase transaction, recommend the best card from the user's "
        "existing portfolio to maximize cashback or reward points."
    )
    max_steps = 1

    def __init__(self, catalogue: List[CreditCard], seed: Optional[int] = None):
        super().__init__(catalogue)
        self._rng = random.Random(seed)
        self._current_user: Optional[UserProfile] = None
        self._current_transaction: Optional[Transaction] = None
        self._step_count = 0
        self._action_history: List[Action] = []
        self._obs_history: List[Observation] = []

    def reset(
        self,
        user: Optional[UserProfile] = None,
        transaction: Optional[Transaction] = None,
    ) -> Observation:  # type: ignore[override]
        self._step_count = 0
        self._action_history = []
        self._obs_history = []

        if user is not None and transaction is not None:
            self._current_user = copy.deepcopy(user)
            self._current_transaction = copy.deepcopy(transaction)
        else:
            chosen_user, chosen_txn = self._rng.choice(_DEFAULT_SCENARIOS)
            self._current_user = copy.deepcopy(chosen_user)
            self._current_transaction = copy.deepcopy(chosen_txn)

        owned = self._current_user.owned_cards
        obs = Observation(
            user_profile=self._current_user,
            available_cards=self.catalogue,
            current_transaction=self._current_transaction,
            step_number=0,
            max_steps=self.max_steps,
            task_name=self.name,
            task_difficulty=self.difficulty,
            context={
                "objective": self.description,
                "instruction": (
                    f"Select the BEST card from {self._current_user.name}'s owned cards to pay for this transaction. "
                    f"Owned cards: {owned}. "
                    f"Transaction: INR {self._current_transaction.amount} at {self._current_transaction.merchant} "
                    f"(category: {self._current_transaction.category}). "
                    f"ONLY recommend cards in the owned list."
                ),
                "owned_cards": owned,
            },
        )
        self._obs_history.append(obs)
        return obs

    def step(
        self, action: Action, observation: Observation
    ) -> tuple[Observation, Reward, bool, Dict[str, Any]]:
        self._step_count += 1
        self._action_history.append(action)

        reward = compute_transaction_reward(
            action=action,
            user=self._current_user,
            transaction=self._current_transaction,
            catalogue=self.catalogue,
        )

        done = self._step_count >= self.max_steps

        next_obs = Observation(
            user_profile=self._current_user,
            available_cards=self.catalogue,
            current_transaction=self._current_transaction,
            step_number=self._step_count,
            max_steps=self.max_steps,
            task_name=self.name,
            task_difficulty=self.difficulty,
            context={
                "last_action": action.recommended_card,
                "last_reward": reward.score,
                "done": done,
            },
        )
        self._obs_history.append(next_obs)
        return next_obs, reward, done, {"step": self._step_count}

    def grade(self, actions: List[Action], observations: List[Observation]) -> float:
        if not actions or self._current_user is None:
            return 0.0

        reward = compute_transaction_reward(
            action=actions[0],
            user=self._current_user,
            transaction=self._current_transaction,
            catalogue=self.catalogue,
        )
        return reward.score
