"""
Abstract base class for all OpenEnv tasks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from credit_card_env.models import (
    Action,
    CreditCard,
    Observation,
    Reward,
    TaskDifficulty,
    UserProfile,
)


class BaseTask(ABC):
    """
    All tasks must inherit from BaseTask and implement the required methods.
    """

    name: str = "base_task"
    difficulty: TaskDifficulty = TaskDifficulty.EASY
    description: str = ""
    max_steps: int = 1

    def __init__(self, catalogue: List[CreditCard]):
        self.catalogue = catalogue

    @abstractmethod
    def reset(self, user: Optional[UserProfile] = None) -> Observation:
        """Reset the task and return the initial observation."""

    @abstractmethod
    def step(self, action: Action, observation: Observation) -> tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Process an action and return (next_obs, reward, done, info).
        """

    @abstractmethod
    def grade(self, actions: List[Action], observations: List[Observation]) -> float:
        """
        Deterministic grader: given the full action/observation history,
        return a final score in [0.0, 1.0].
        """
