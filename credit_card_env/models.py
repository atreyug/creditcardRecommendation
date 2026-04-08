"""
Pydantic models for the Credit Card Recommendation OpenEnv environment.
All domain objects are strictly typed with validation.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


# ---------------------------------------------------------------------------
# Domain enumerations
# ---------------------------------------------------------------------------

class SpendingCategory(str, Enum):
    DINING = "dining"
    TRAVEL = "travel"
    GROCERIES = "groceries"
    FUEL = "fuel"
    ONLINE = "online"
    ENTERTAINMENT = "entertainment"
    GENERAL = "general"


class CreditScoreTier(str, Enum):
    POOR = "poor"          # < 600
    FAIR = "fair"          # 600–699
    GOOD = "good"          # 700–749
    VERY_GOOD = "very_good"  # 750–799
    EXCELLENT = "excellent"  # 800+


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ---------------------------------------------------------------------------
# Credit Card domain object
# ---------------------------------------------------------------------------

class CreditCard(BaseModel):
    """Represents a single credit card product."""

    card_name: str = Field(..., description="Full product name of the card")
    bank_name: str = Field(..., description="Issuing bank or financial institution")
    joining_fee: float = Field(0.0, ge=0.0, description="One-time joining / activation fee in INR")
    annual_fee: float = Field(0.0, ge=0.0, description="Annual maintenance fee in INR")
    welcome_offer: str = Field("", description="Description of the welcome bonus")
    primary_category: SpendingCategory = Field(
        SpendingCategory.GENERAL, description="Primary category this card excels in"
    )
    categories: List[SpendingCategory] = Field(
        default_factory=list, description="All supported reward categories"
    )
    benefits: List[str] = Field(default_factory=list, description="List of card benefits")
    reward_rate: float = Field(
        1.0, ge=0.0, le=100.0,
        description="Base reward / cashback rate as a percentage"
    )
    annual_benefit_value: float = Field(
        0.0, ge=0.0,
        description="Estimated total annual benefit value in INR"
    )
    min_income_required: float = Field(
        0.0, ge=0.0,
        description="Minimum annual income required to be eligible in INR"
    )
    credit_score_required: CreditScoreTier = Field(
        CreditScoreTier.GOOD, description="Minimum credit score tier required"
    )

    class Config:
        use_enum_values = True

    @property
    def net_annual_value(self) -> float:
        """Annual benefit minus annual fee."""
        return self.annual_benefit_value - self.annual_fee

    def category_reward_rate(self, category: SpendingCategory) -> float:
        """Return effective reward rate for a given spending category."""
        if category == self.primary_category:
            return self.reward_rate * 2.0          # 2× for primary category
        if category in self.categories:
            return self.reward_rate * 1.5           # 1.5× for supported categories
        return self.reward_rate                     # base rate for everything else


# ---------------------------------------------------------------------------
# User Profile
# ---------------------------------------------------------------------------

class UserProfile(BaseModel):
    """Represents the financial profile of the user interacting with the agent."""

    user_id: str = Field(..., description="Unique user identifier")
    name: str = Field("", description="User's name")
    annual_income: float = Field(
        500_000.0, ge=0.0, description="Annual income in INR"
    )
    credit_score_tier: CreditScoreTier = Field(
        CreditScoreTier.GOOD, description="User's current credit score tier"
    )
    owned_cards: List[str] = Field(
        default_factory=list, description="List of card_names the user already owns"
    )
    monthly_spending: Dict[str, float] = Field(
        default_factory=dict,
        description="Monthly spending in INR per SpendingCategory key"
    )

    class Config:
        use_enum_values = True

    @property
    def top_spending_category(self) -> SpendingCategory:
        """Return the category with highest monthly spend."""
        if not self.monthly_spending:
            return SpendingCategory.GENERAL
        top = max(self.monthly_spending, key=lambda k: self.monthly_spending[k])
        try:
            return SpendingCategory(top)
        except ValueError:
            return SpendingCategory.GENERAL

    @property
    def total_monthly_spend(self) -> float:
        return sum(self.monthly_spending.values())


# ---------------------------------------------------------------------------
# Transaction
# ---------------------------------------------------------------------------

class Transaction(BaseModel):
    """A single financial transaction to be optimized."""

    merchant: str = Field("", description="Merchant name")
    amount: float = Field(..., gt=0.0, description="Transaction amount in INR")
    category: SpendingCategory = Field(
        SpendingCategory.GENERAL, description="Spending category of the transaction"
    )

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# OpenEnv core types
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    The full observable state returned to the agent at each step.
    Contains user profile, available card database and optional transaction context.
    """

    user_profile: UserProfile
    available_cards: List[CreditCard] = Field(
        default_factory=list,
        description="Full catalogue of credit cards the agent can recommend"
    )
    current_transaction: Optional[Transaction] = Field(
        None, description="Transaction to optimize (MEDIUM / HARD tasks)"
    )
    step_number: int = Field(0, ge=0, description="Current step within episode")
    max_steps: int = Field(1, ge=1, description="Total steps in the episode")
    task_name: str = Field("", description="Name of the active task")
    task_difficulty: TaskDifficulty = Field(TaskDifficulty.EASY)
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Task-specific extra context"
    )

    class Config:
        use_enum_values = True


class Action(BaseModel):
    """
    Action taken by the agent — a card recommendation.
    """

    recommended_card: str = Field(
        ..., description="Exact card_name to recommend / use for the transaction"
    )
    reasoning: str = Field(
        "", description="Optional natural-language reasoning from the agent"
    )

    class Config:
        use_enum_values = True


class RewardBreakdown(BaseModel):
    """Granular reward component scores summing to the total."""

    category_match: float = Field(0.0, description="Score for category alignment")
    reward_rate: float = Field(0.0, description="Score for reward/cashback rate")
    fee_efficiency: float = Field(0.0, description="Score for fee vs benefit ratio")
    novelty_bonus: float = Field(0.0, description="Bonus for non-redundant recommendation")
    welcome_offer_bonus: float = Field(0.0, description="Bonus for attractive welcome offer")
    eligibility_penalty: float = Field(0.0, description="Penalty for ineligible card")
    redundancy_penalty: float = Field(0.0, description="Penalty for recommending owned card")


class Reward(BaseModel):
    """
    Reward signal returned after each step.
    Score is always in [0.0, 1.0].
    """

    score: float = Field(..., ge=0.0, le=1.0, description="Total reward score")
    breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
    rationale: str = Field("", description="Human-readable explanation of the reward")
    action_taken: str = Field("", description="Card that was recommended")

    @validator("score", pre=True)
    def clamp_score(cls, v: float) -> float:  # noqa: N805
        return max(0.0, min(1.0, float(v)))


class StepResult(BaseModel):
    """Full result of a step() call — mirrors the OpenEnv (obs, reward, done, info) tuple."""

    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
