---
title: Credit Card Recommendation AI
emoji: 💳
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---
<<<<<<< HEAD
# 💳 Credit Card Recommendation AI — Multi-Task Environment

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT_4o_mini-green)](https://openai.com/)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-orange)](https://openenv.ai/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)

A production-grade, highly sophisticated environment built to benchmark AI agents on **Credit Card Recommendations, Transaction Routing, and Portfolio Management**. 

It uses a dynamic dataset of real-world Indian credit cards, combining both offline analytical constraints and a **True AI LLM-based Advisor** interface that reads unstructured complex inputs to perfectly route financial decisions.

---

## 📖 Environment Description

Credit card selection is a complex multidimensional optimization problem. Users navigate hundreds of options, fee-vs-benefit tradeoffs, changing lifestyle contexts, and eligibility constraints. 

This environment simulates a **Financial AI Advisor** acting as an Agent to solve progressively complex financial pathways:
- **Phase 1:** Recommend one highly optimized card based on user income and lifestyle.
- **Phase 2:** Given a custom portfolio and a stream of user transactions (e.g. Swiggy dining, MakeMyTrip flights, IndianOil fuel), route each transaction to the *exact* owned card that yields strictly the highest reward rate.

The Environment ensures all AI agents learn rigorous adherence to real-world financial heuristics through a robust, 6-component penalization and reward structure. 

---

## 🔍 Observation Space

At each step, your Agent receives a `state()` containing a highly detailed context representing the current frame of the financial simulation.

| Field | Type | Description |
|-------|------|-------------|
| `user_profile.annual_income` | `float` | The simulated user's income in INR. |
| `user_profile.credit_score_tier` | `enum` | Categorical tier (e.g., `good`, `excellent`). |
| `user_profile.owned_cards` | `List[str]` | The current credit cards active in the user's portfolio. |
| `user_profile.monthly_spending` | `Dict` | Breakdown of monthly category spends. |
| `available_cards` | `List[Card]` | The entire database of credit cards and all meta-attributes. |
| `current_transaction` | `Transaction` | If running a routing task, the exact transaction being optimized. |
| `task_name` | `str` | Active task identifier (e.g. `portfolio_optimization_hard`). |
| `context` | `Dict` | Granular active-phase instructions. |

---

## 🕹️ Action Space

The Agent must respond to the `step()` function by predicting exactly one structured response block. 

```python
class Action(BaseModel):
    recommended_card: str   # MUST match an exact 'card_name' string from available_cards
    reasoning: str = ""     # Human-readable rationale for debugging and transparency
```

---

## 🏆 Reward & Grading Mechanism

Agents are graded continuously. The internal grader validates every `step()` execution and provides a continuous `reward ∈ [0.0, 1.0]`.

| Component | Weight | Rule |
|-----------|--------|------|
| **Category Alignment** | +30% | Card perfectly covers the targeted transaction/lifestyle. |
| **Reward Rate** | +30% | Achieves the mathematical maximum cashback % available. |
| **Fee Efficiency** | +15% | The card carries high-value perks compared to annual maintenance. |
| **Novelty / Discovery** | +15% | Properly recommends an unowned asset where required. |
| **Welcome Bonus** | +10% | Secures strong introductory signup rewards. |
| **Eligibility Penalty** | -40% | *Penalty* for recommending a card the user doesn't physically qualify for. |
| **Redundancy Penalty**| -25% | *Penalty* for recommending an already owned asset incorrectly. |

---

## 🗂️ Task Difficulty Matrix

### 1. `card_recommendation_easy`
* **Objective:** Give a user their very first tailored Credit card based purely on static lifestyle indicators.
* **Steps:** 1
* **Constraint:** Card must not already be in `owned_cards`.

### 2. `transaction_optimization_medium`
* **Objective:** The user is making a purchase right now (e.g., "₹450 on Zomato"). Select the optimal card to use from their current wallet.
* **Steps:** 1
* **Constraint:** The recommendation MUST be a card inside `owned_cards`.

### 3. `portfolio_optimization_hard`
* **Objective:** A longitudinal study. First, recommend one net-new card to permanently add to the user's wallet. Second, process a chronological sequence of 5 distinct dynamic transactions and route each one perfectly. 
* **Steps:** 6 consecutive turns.

---

## 🚀 Setup Instructions

### 1. Prerequisites
- **Python:** 3.10 or higher.
- **OpenAI API Key:** Required to power the True AI interactive engine.

### 2. Installation
Clone the required repository and initialize the Python environment:

```bash
# 1. Enter the project directory
cd metaHackathon

# 2. Setup your virtual environment
python -m venv .venv

# 3. Activate the environment
# For Windows:
.venv\Scripts\activate
# For Mac/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 3. Configuration & Database
You **must** configure your `.env` securely in the root directory before launching:

```env
OPENAI_API_KEY=sk-proj-your-api-key
MODEL_NAME=gpt-4o-mini
API_BASE_URL=https://api.openai.com/v1
```

*(Note: The environment will automatically parse any custom `credit_cards.xlsx` you place into the `data/` directory. If none exists, it gracefully falls back to a synthesized, normalized array of 15 standard cards).*

---

## 💻 Operating the Environment

There are three distinct layers to the Credit Card AI environment depending on your testing structure.

### 1. True AI Natural Language Interface (`ask.py`)
Chat dynamically with the underlying Agent. It receives the entire database structurally, understands deep context, and computes real-time recommendations.

```bash
python ask.py
```
> **Prompt Example:** *"I want a free card for swiggy"*  
> **Backend Response:** Passes deep system constraints to `gpt-4o-mini`, maps exactly to lifetime-free Zomato/Swiggy optimized tier cards.

*(Requires active OpenAI API Quota).*

### 2. Full Benchmark Inference (`inference.py`)
Programmatically iterate over all three difficulty subsets (`Easy`, `Medium`, `Hard`) to receive an exact heuristic benchmark grade comparing the baseline AI to the internal OpenEnv constraints.

```bash
python inference.py
```

### 3. Production FastAPI Local Server (`app.py`)
Expose the OpenEnv classes visually via Swagger UI over localhost for containerized frontend linkage.

```bash
python app.py
# Server connects to http://localhost:8000
# OpenAPI Swagger available at http://localhost:8000/docs
```

---

## 🐳 Docker Production (Optional)

Run the environment seamlessly on cloud services.

```bash
docker build -t credit-card-env:latest .
docker run -p 7860:7860 --env-file .env credit-card-env:latest
```

This ensures portability on Hugging Face Spaces or AWS EC2 without local Python overhead.
=======
---
title: CreditCardRecommendation
emoji: 🌖
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.11.0
app_file: app.py
pinned: false
short_description: My project on Credit Card Recommendation system.
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 5d33c5abad98fbe6a58c25963eb91a5a9de7fcd5
