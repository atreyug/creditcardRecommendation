"""
EASY Task: Card Recommendation from User Profile.

Objective:
    Given a user profile (income, credit score, spending habits),
    recommend the single best NEW credit card from the catalogue.

Input State:
    - UserProfile with monthly_spending and credit score
    - Full credit card catalogue
    - No current transaction

Expected Behaviour:
    - Recommend a card whose primary category matches the user's top spend
    - Card must be new (not already owned)
    - Card should be fee-efficient for the user's income level

Grader:
    Deterministic score in [0.0, 1.0] based on:
    - Category match (30%)
    - Reward rate (30%)
    - Fee efficiency (15%)
    - Novelty / welcome offer (25%)
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
    UserProfile,
)
from credit_card_env.reward import compute_recommendation_reward
from credit_card_env.tasks.base import BaseTask

# ---------------------------------------------------------------------------
# Default user profiles to sample from during resets
# ---------------------------------------------------------------------------
_DEFAULT_PROFILES = [
    UserProfile(
        user_id="u_easy_01",
        name="Priya Sharma",
        annual_income=600_000.0,
        credit_score_tier=CreditScoreTier.GOOD,
        owned_cards=["ICICI Amazon Pay"],
        monthly_spending={
            SpendingCategory.ONLINE: 8000,
            SpendingCategory.DINING: 4000,
            SpendingCategory.GROCERIES: 3000,
            SpendingCategory.FUEL: 2000,
        },
    ),
    UserProfile(
        user_id="u_easy_02",
        name="Rahul Mehta",
        annual_income=450_000.0,
        credit_score_tier=CreditScoreTier.GOOD,
        owned_cards=[],
        monthly_spending={
            SpendingCategory.FUEL: 6000,
            SpendingCategory.GROCERIES: 4000,
            SpendingCategory.GENERAL: 3000,
        },
    ),
    UserProfile(
        user_id="u_easy_03",
        name="Sneha Kapoor",
        annual_income=900_000.0,
        credit_score_tier=CreditScoreTier.VERY_GOOD,
        owned_cards=["HDFC MoneyBack+"],
        monthly_spending={
            SpendingCategory.TRAVEL: 15000,
            SpendingCategory.DINING: 8000,
            SpendingCategory.ENTERTAINMENT: 4000,
        },
    ),
]


class CardRecommendationTask(BaseTask):
    """
    EASY — Single-step card recommendation from user profile.
    """

    name = "card_recommendation_easy"
    difficulty = TaskDifficulty.EASY
    description = (
        "Recommend the best new credit card for a user based on their spending habits, "
        "credit score tier, and income. The card must not already be owned by the user."
    )
    max_steps = 1

    def __init__(self, catalogue: List[CreditCard], seed: Optional[int] = None):
        super().__init__(catalogue)
        self._rng = random.Random(seed)
        self._current_user: Optional[UserProfile] = None
        self._step_count = 0
        self._action_history: List[Action] = []
        self._obs_history: List[Observation] = []

    def reset(self, user: Optional[UserProfile] = None) -> Observation:
        self._step_count = 0
        self._action_history = []
        self._obs_history = []

        if user is not None:
            self._current_user = copy.deepcopy(user)
        else:
            self._current_user = copy.deepcopy(self._rng.choice(_DEFAULT_PROFILES))

        obs = Observation(
            user_profile=self._current_user,
            available_cards=self.catalogue,
            current_transaction=None,
            step_number=0,
            max_steps=self.max_steps,
            task_name=self.name,
            task_difficulty=self.difficulty,
            context={
                "objective": self.description,
                "instruction": (
                    f"Recommend the single best NEW credit card for {self._current_user.name}. "
                    f"Top spending category: {self._current_user.top_spending_category}. "
                    f"Do NOT recommend cards they already own: {self._current_user.owned_cards}."
                ),
            },
        )
        self._obs_history.append(obs)
        return obs

    def step(
        self, action: Action, observation: Observation
    ) -> tuple[Observation, Reward, bool, Dict[str, Any]]:
        self._step_count += 1
        self._action_history.append(action)

        reward = compute_recommendation_reward(
            action=action,
            user=self._current_user,
            catalogue=self.catalogue,
            target_category=self._current_user.top_spending_category,
            require_new_card=True,
        )

        done = self._step_count >= self.max_steps

        next_obs = Observation(
            user_profile=self._current_user,
            available_cards=self.catalogue,
            current_transaction=None,
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

        return next_obs, reward, done, {"step": self._step_count, "reward_breakdown": reward.breakdown.dict()}

    def grade(self, actions: List[Action], observations: List[Observation]) -> float:
        """
        Deterministic grader.
        Returns the reward score of the first (and only) action taken,
        or 0.0 if no action was taken.
        """
        if not actions or self._current_user is None:
            return 0.0

        reward = compute_recommendation_reward(
            action=actions[0],
            user=self._current_user,
            catalogue=self.catalogue,
            target_category=self._current_user.top_spending_category,
            require_new_card=True,
        )
        return reward.score
