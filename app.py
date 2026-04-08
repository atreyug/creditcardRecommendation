from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict
import os

from credit_card_env import CreditCardEnv, Action

# -------------------------------------------------------
# App setup
# -------------------------------------------------------
app = FastAPI(title="OpenEnv Credit Card API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# Global environment
# -------------------------------------------------------
_ENV = None
EXCEL_PATH = os.getenv("EXCEL_PATH", None)


def get_env():
    global _ENV
    if _ENV is None:
        _ENV = CreditCardEnv(excel_path=EXCEL_PATH, seed=42)
    return _ENV


# -------------------------------------------------------
# Root + Health
# -------------------------------------------------------
@app.get("/")
def root():
    return {"status": "running"}


@app.get("/health")
def health():
    env = get_env()
    return {
        "status": "ok",
        "env_id": env.ENVIRONMENT_ID,
        "version": env.VERSION,
    }


# -------------------------------------------------------
# 🔥 CRITICAL FIX: RESET (NO INPUT)
# -------------------------------------------------------
@app.post("/reset")
def reset():
    """
    MUST accept empty POST request (OpenEnv requirement)
    """
    env = get_env()

    try:
        obs = env.reset(task_name="easy")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": obs.dict(),
        "task": "easy",
        "available_cards": env.available_cards,
    }


# -------------------------------------------------------
# STEP
# -------------------------------------------------------
@app.post("/step")
def step(data: Dict[str, Any]):
    env = get_env()

    if env._current_task is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    if env._done:
        raise HTTPException(status_code=400, detail="Episode done")

    try:
        action = Action(
            recommended_card=data.get("recommended_card", ""),
            reasoning=data.get("reasoning", ""),
        )
        obs, reward, done, info = env.step(action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "info": info,
    }


# -------------------------------------------------------
# STATE + GRADE
# -------------------------------------------------------
@app.get("/state")
def state():
    return get_env().state()


@app.get("/grade")
def grade():
    return {"grade": get_env().grade()}
