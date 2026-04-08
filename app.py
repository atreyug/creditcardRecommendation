"""
FastAPI application — exposes the CreditCardEnv over HTTP.
Compatible with Hugging Face Spaces (Docker SDK).

Endpoints:
    POST /reset         → { observation: ..., task: ... }
    POST /step          → { observation, reward, done, info }
    GET  /state         → current state dict
    GET  /health        → { status: "ok", ... }
    GET  /tasks         → list of available tasks
    GET  /cards         → list of available credit cards
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from credit_card_env import CreditCardEnv, Action
from credit_card_env.models import Observation, Reward, StepResult, UserProfile

# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Credit Card Recommendation OpenEnv API",
    description=(
        "Production-grade OpenEnv environment for AI-driven credit card recommendation. "
        "Supports 3 difficulty tasks: Easy, Medium, Hard."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global singleton environment (per-process)
_ENV: Optional[CreditCardEnv] = None

EXCEL_PATH = os.getenv("EXCEL_PATH", None)


def get_env() -> CreditCardEnv:
    global _ENV
    if _ENV is None:
        _ENV = CreditCardEnv(excel_path=EXCEL_PATH, seed=42)
    return _ENV


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: str = "easy"
    user: Optional[Dict[str, Any]] = None


class StepRequest(BaseModel):
    recommended_card: str
    reasoning: str = ""


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    task: str
    available_cards: list


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    env = get_env()
    return {
        "status": "ok",
        "environment_id": env.ENVIRONMENT_ID,
        "version": env.VERSION,
        "catalogue_size": len(env.catalogue),
        "available_tasks": list(env.available_tasks.keys()),
    }


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    env = get_env()
    return {
        "tasks": [
            {
                "name": name,
                "difficulty": difficulty,
            }
            for name, difficulty in env.available_tasks.items()
        ]
    }


@app.get("/cards")
def list_cards() -> Dict[str, Any]:
    env = get_env()
    return {
        "count": len(env.catalogue),
        "cards": [
            {
                "card_name": card.card_name,
                "bank_name": card.bank_name,
                "primary_category": card.primary_category,
                "annual_fee": card.annual_fee,
                "reward_rate": card.reward_rate,
            }
            for card in env.catalogue
        ],
    }


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest) -> ResetResponse:
    """Reset the environment for a specific task."""
    env = get_env()
    try:
        user = UserProfile(**req.user) if req.user else None
        obs = env.reset(task_name=req.task_name, user=user)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return ResetResponse(
        observation=obs.dict(),
        task=req.task_name,
        available_cards=env.available_cards,
    )


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    """Take a step — provide a card recommendation."""
    env = get_env()
    if env._current_task is None:
        raise HTTPException(status_code=400, detail="Call /reset before /step")
    if env._done:
        raise HTTPException(status_code=400, detail="Episode done. Call /reset to start a new episode.")

    action = Action(recommended_card=req.recommended_card, reasoning=req.reasoning)
    try:
        next_obs, reward, done, info = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return StepResponse(
        observation=next_obs.dict(),
        reward=reward.dict(),
        done=done,
        info=info,
    )


@app.get("/state")
def state() -> Dict[str, Any]:
    """Return the current environment state."""
    env = get_env()
    return env.state()


@app.get("/grade")
def grade() -> Dict[str, float]:
    """Return the deterministic grade for the current episode."""
    env = get_env()
    return {"grade": env.grade()}
