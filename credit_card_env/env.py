"""
CreditCardEnv — Main OpenEnv environment.

Implements the full OpenEnv interface:
    reset()  → Observation
    step()   → (Observation, Reward, done, info)
    state()  → Dict[str, Any]

Tasks are registered by name and can be switched via reset(task_name=...).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

from credit_card_env.data_loader import load_cards
from credit_card_env.models import (
    Action,
    CreditCard,
    Observation,
    Reward,
    StepResult,
    TaskDifficulty,
    UserProfile,
)
from credit_card_env.tasks.base import BaseTask
from credit_card_env.tasks.easy import CardRecommendationTask
from credit_card_env.tasks.hard import PortfolioOptimizationTask
from credit_card_env.tasks.medium import TransactionOptimizationTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
_TASK_REGISTRY: Dict[str, Type[BaseTask]] = {
    CardRecommendationTask.name: CardRecommendationTask,
    TransactionOptimizationTask.name: TransactionOptimizationTask,
    PortfolioOptimizationTask.name: PortfolioOptimizationTask,
    # convenience aliases
    "easy": CardRecommendationTask,
    "medium": TransactionOptimizationTask,
    "hard": PortfolioOptimizationTask,
}


class CreditCardEnv:
    """
    Credit Card Recommendation OpenEnv.

    Usage:
        env = CreditCardEnv()
        obs = env.reset(task_name="easy")
        action = Action(recommended_card="HDFC Millennia")
        obs, reward, done, info = env.step(action)
    """

    VERSION = "1.0.0"
    ENVIRONMENT_ID = "credit-card-recommendation-v1"

    def __init__(
        self,
        excel_path: Optional[str] = None,
        seed: Optional[int] = 42,
    ):
        self.catalogue: List[CreditCard] = load_cards(excel_path)
        self._seed = seed
        self._current_task: Optional[BaseTask] = None
        self._current_obs: Optional[Observation] = None
        self._done: bool = False
        self._step_count: int = 0
        self._action_history: List[Action] = []
        self._obs_history: List[Observation] = []
        self._reward_history: List[Reward] = []

        logger.info(
            "CreditCardEnv initialised with %d cards. Seed=%s",
            len(self.catalogue),
            seed,
        )

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        task_name: str = "easy",
        user: Optional[UserProfile] = None,
    ) -> Observation:
        """
        Reset the environment and return the initial observation.

        Args:
            task_name: One of 'easy', 'medium', 'hard' or the full task name.
            user: Optional custom UserProfile; if None, a default is selected.

        Returns:
            Initial Observation.
        """
        task_cls = _TASK_REGISTRY.get(task_name)
        if task_cls is None:
            available = list(_TASK_REGISTRY.keys())
            raise ValueError(
                f"Unknown task '{task_name}'. Available tasks: {available}"
            )

        self._current_task = task_cls(catalogue=self.catalogue, seed=self._seed)
        self._done = False
        self._step_count = 0
        self._action_history = []
        self._obs_history = []
        self._reward_history = []

        if user is not None:
            obs = self._current_task.reset(user=user)
        else:
            obs = self._current_task.reset()

        self._current_obs = obs
        self._obs_history.append(obs)
        logger.info("Environment reset for task '%s'.", task_name)
        return obs

    def step(self, action: Action) -> tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: The agent's Action (recommended card + optional reasoning).

        Returns:
            Tuple of (Observation, Reward, done, info).

        Raises:
            RuntimeError: If called before reset() or after episode end.
        """
        if self._current_task is None:
            raise RuntimeError("Call env.reset() before env.step().")
        if self._done:
            raise RuntimeError("Episode is done. Call env.reset() to start a new episode.")

        next_obs, reward, done, info = self._current_task.step(action, self._current_obs)

        self._action_history.append(action)
        self._obs_history.append(next_obs)
        self._reward_history.append(reward)
        self._current_obs = next_obs
        self._done = done
        self._step_count += 1

        total_so_far = sum(r.score for r in self._reward_history)
        info["cumulative_reward"] = total_so_far
        info["episode_step"] = self._step_count

        return next_obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        """
        Return the current environment state as a serialisable dict.
        """
        return {
            "environment_id": self.ENVIRONMENT_ID,
            "version": self.VERSION,
            "task": self._current_task.name if self._current_task else None,
            "step": self._step_count,
            "done": self._done,
            "current_observation": self._current_obs.dict() if self._current_obs else None,
            "action_history": [a.dict() for a in self._action_history],
            "reward_history": [r.dict() for r in self._reward_history],
            "total_reward": sum(r.score for r in self._reward_history),
            "catalogue_size": len(self.catalogue),
        }

    # ------------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------------

    def grade(self) -> float:
        """
        Return the final deterministic grade for the episode.
        """
        if self._current_task is None:
            return 0.0
        return self._current_task.grade(self._action_history, self._obs_history)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def available_tasks(self) -> Dict[str, str]:
        """Return a mapping of task_name → difficulty for all registered tasks."""
        seen: Dict[str, str] = {}
        for name, cls in _TASK_REGISTRY.items():
            if cls.name not in seen:
                seen[cls.name] = cls.difficulty
        return seen

    @property
    def available_cards(self) -> List[str]:
        """Return list of card names in the catalogue."""
        return [c.card_name for c in self.catalogue]

    def __repr__(self) -> str:
        task = self._current_task.name if self._current_task else "none"
        return (
            f"CreditCardEnv(task={task!r}, step={self._step_count}, "
            f"done={self._done}, cards={len(self.catalogue)})"
        )
