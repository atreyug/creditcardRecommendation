from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from credit_card_env import CreditCardEnv, Action
from credit_card_env.models import UserProfile

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Credit Card Recommendation OpenEnv API",
    description="OpenEnv environment for AI-driven credit card recommendation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global Environment
# ---------------------------------------------------------------------------
_ENV: Optional[CreditCardEnv] = None
EXCEL_PATH = os.getenv("EXCEL_PATH", None)


def get_env() -> CreditCardEnv:
    global _ENV
    if _ENV is None:
        _ENV = CreditCardEnv(excel_path=EXCEL_PATH, seed=42)
    return _ENV


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    task_name: str = "easy"
    user: Optional[Dict[str, Any]] = None


class StepRequest(BaseModel):
    recommended_card: str
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Basic Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "running"}


@app.get("/health")
def health():
    env = get_env()
    return {
        "status": "ok",
        "environment_id": env.ENVIRONMENT_ID,
        "version": env.VERSION,
    }


@app.get("/tasks")
def tasks():
    env = get_env()
    return {"tasks": list(env.available_tasks.keys())}


@app.get("/cards")
def cards():
    env = get_env()
    return {"cards": [card.card_name for card in env.catalogue]}


# ---------------------------------------------------------------------------
# FIXED RESET ENDPOINT (IMPORTANT)
# ---------------------------------------------------------------------------
@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    """
    Works with or WITHOUT request body (OpenEnv requirement)
    """
    env = get_env()

    # Default request if none provided
    if req is None:
        req = ResetRequest()

    try:
        user = UserProfile(**req.user) if req.user else None
        obs = env.reset(task_name=req.task_name, user=user)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": obs.dict(),
        "task": req.task_name,
        "available_cards": env.available_cards,
    }


# ---------------------------------------------------------------------------
# STEP ENDPOINT
# ---------------------------------------------------------------------------
@app.post("/step")
def step(req: StepRequest):
    env = get_env()

    if env._current_task is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    if env._done:
        raise HTTPException(status_code=400, detail="Episode finished. Call /reset")

    action = Action(
        recommended_card=req.recommended_card,
        reasoning=req.reasoning
    )

    try:
        next_obs, reward, done, info = env.step(action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": next_obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "info": info,
    }


# ---------------------------------------------------------------------------
# STATE + GRADE
# ---------------------------------------------------------------------------
@app.get("/state")
def state():
    env = get_env()
    return env.state()


@app.get("/grade")
def grade():
    env = get_env()
    return {"grade": env.grade()}
