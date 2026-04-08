"""
Data loader: reads the Excel credit card dataset and converts it into
structured CreditCard Pydantic objects. Falls back to a rich synthetic
seed dataset if the Excel file is absent.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

from credit_card_env.models import CreditCard, CreditScoreTier, SpendingCategory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category keyword mapping (used to infer categories from text)
# ---------------------------------------------------------------------------
_CATEGORY_KEYWORDS: dict[SpendingCategory, list[str]] = {
    SpendingCategory.DINING: ["dining", "restaurant", "food", "eating", "zomato", "swiggy"],
    SpendingCategory.TRAVEL: ["travel", "flight", "hotel", "airline", "lounge", "airport", "booking", "indigo"],
    SpendingCategory.GROCERIES: ["grocer", "supermarket", "big bazaar", "dmart", "grocery"],
    SpendingCategory.FUEL: ["fuel", "petrol", "diesel", "gas station", "hp", "iocl", "bpcl"],
    SpendingCategory.ONLINE: ["online", "amazon", "flipkart", "myntra", "e-commerce", "shopping"],
    SpendingCategory.ENTERTAINMENT: ["entertainment", "movie", "ott", "netflix", "spotify", "gaming"],
}


def _infer_category(text: str) -> SpendingCategory:
    """Infer the primary spending category from free-form benefit text."""
    text_lower = text.lower()
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return cat
    return SpendingCategory.GENERAL


def _infer_categories(benefits: list[str], welcome: str, primary: SpendingCategory) -> list[SpendingCategory]:
    """Return all supported categories inferred from benefit descriptions."""
    found: set[SpendingCategory] = {primary}
    combined = " ".join(benefits) + " " + welcome
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in combined.lower() for kw in keywords):
            found.add(cat)
    return list(found)


def _safe_float(value, default: float = 0.0) -> float:
    """Convert a potentially messy value to float, stripping INR symbols."""
    if value is None:
        return default
    try:
        cleaned = str(value).replace("₹", "").replace(",", "").replace("Rs.", "").strip()
        cleaned = cleaned.split("/")[0].strip()  # handle "2000/year"
        return float(cleaned)
    except (ValueError, AttributeError):
        return default


def _estimate_annual_benefit(card_name: str, benefits: list[str], reward_rate: float, welcome: str) -> float:
    """
    Heuristic: estimate annual benefit value from reward rate + number of perks.
    Assumes INR 1L annual spend as base and adds flat perks value.
    """
    # Base cashback on average monthly spend of 10K
    base = (reward_rate / 100) * 120_000
    # Add fixed value per named benefit
    perk_value = len(benefits) * 500
    # Add welcome offer value (rough estimate)
    if "bonus" in welcome.lower() or "points" in welcome.lower():
        base += 1500
    if "voucher" in welcome.lower() or "cashback" in welcome.lower():
        base += 2000
    return round(base + perk_value, 2)


# ---------------------------------------------------------------------------
# Excel loader
# ---------------------------------------------------------------------------

def load_from_excel(path: str | Path) -> List[CreditCard]:
    """
    Load credit card data from an Excel (.xlsx) file.

    Expected columns (case-insensitive):
        Card Name, Bank Name, Joining Fee, Maintenance Fees (Annual),
        Welcome Offer, Category of Offer, All the Benefits
    """
    try:
        import openpyxl  # noqa: F401
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "openpyxl and pandas are required to load Excel files. "
            "Install with: pip install pandas openpyxl"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    df = pd.read_excel(path, engine="openpyxl")

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_").replace("(", "").replace(")", "") for c in df.columns]

    col_map = {
        "card_name": ["card_name", "card name", "name"],
        "bank_name": ["bank_name", "bank name", "bank", "issuer"],
        "joining_fee": ["joining_fee", "joining fee", "joining"],
        "annual_fee": ["maintenance_fees_annual", "annual_fee", "annual fee", "maintenance fee"],
        "welcome_offer": ["welcome_offer", "welcome offer"],
        "category": ["category_of_offer", "category of offer", "category"],
        "benefits": ["all_the_benefits", "benefits", "all benefits"],
    }

    def find_col(candidates: list[str]) -> str | None:
        for c in candidates:
            normalised = c.lower().replace(" ", "_").replace("(", "").replace(")", "")
            if normalised in df.columns:
                return normalised
        return None

    resolved = {k: find_col(v) for k, v in col_map.items()}

    cards: List[CreditCard] = []
    for _, row in df.iterrows():
        def get(field: str, default="") -> str:
            col = resolved.get(field)
            if col and col in row.index and pd.notna(row[col]):
                return str(row[col]).strip()
            return str(default)

        card_name = get("card_name", f"Card_{_}")
        bank_name = get("bank_name", "Unknown Bank")
        joining_fee = _safe_float(get("joining_fee", 0))
        annual_fee = _safe_float(get("annual_fee", 0))
        welcome_offer = get("welcome_offer", "")
        category_text = get("category", "general")
        benefits_raw = get("benefits", "")

        benefits = [b.strip() for b in benefits_raw.split(";") if b.strip()]
        if not benefits and benefits_raw:
            benefits = [b.strip() for b in benefits_raw.split("\n") if b.strip()]

        primary_category = _infer_category(category_text + " " + benefits_raw)
        all_categories = _infer_categories(benefits, welcome_offer, primary_category)

        # Reward rate from category text
        import re
        rate_match = re.search(r"(\d+(?:\.\d+)?)(?:\s*%|\s*percent|\s*x)", category_text + " " + benefits_raw, re.IGNORECASE)
        reward_rate = float(rate_match.group(1)) if rate_match else 1.5

        annual_benefit = _estimate_annual_benefit(card_name, benefits, reward_rate, welcome_offer)

        card = CreditCard(
            card_name=card_name,
            bank_name=bank_name,
            joining_fee=joining_fee,
            annual_fee=annual_fee,
            welcome_offer=welcome_offer,
            primary_category=primary_category,
            categories=all_categories,
            benefits=benefits,
            reward_rate=min(reward_rate, 20.0),
            annual_benefit_value=annual_benefit,
            min_income_required=_infer_min_income(annual_fee),
            credit_score_required=_infer_credit_score(annual_fee),
        )
        cards.append(card)

    logger.info("Loaded %d credit cards from %s", len(cards), path)
    return cards


def _infer_min_income(annual_fee: float) -> float:
    """Estimate minimum income from annual fee tier."""
    if annual_fee == 0:
        return 0
    if annual_fee <= 500:
        return 300_000
    if annual_fee <= 2000:
        return 500_000
    if annual_fee <= 5000:
        return 800_000
    return 1_200_000


def _infer_credit_score(annual_fee: float) -> CreditScoreTier:
    if annual_fee == 0:
        return CreditScoreTier.FAIR
    if annual_fee <= 2000:
        return CreditScoreTier.GOOD
    if annual_fee <= 5000:
        return CreditScoreTier.VERY_GOOD
    return CreditScoreTier.EXCELLENT


# ---------------------------------------------------------------------------
# Synthetic seed dataset (fallback)
# ---------------------------------------------------------------------------

def load_synthetic_cards() -> List[CreditCard]:
    """Return a rich synthetic dataset of 15 Indian credit cards for development/testing."""
    return [
        CreditCard(
            card_name="HDFC Regalia",
            bank_name="HDFC Bank",
            joining_fee=2500.0,
            annual_fee=2500.0,
            welcome_offer="2500 reward points on joining",
            primary_category=SpendingCategory.TRAVEL,
            categories=[SpendingCategory.TRAVEL, SpendingCategory.DINING, SpendingCategory.ONLINE],
            benefits=[
                "4 reward points per INR 150 on travel",
                "2 complimentary airport lounge visits per quarter",
                "10% discount on Zomato orders",
                "1% fuel surcharge waiver",
                "Travel insurance cover up to INR 50 lakh",
            ],
            reward_rate=2.67,
            annual_benefit_value=12000.0,
            min_income_required=800_000.0,
            credit_score_required=CreditScoreTier.VERY_GOOD,
        ),
        CreditCard(
            card_name="HDFC MoneyBack+",
            bank_name="HDFC Bank",
            joining_fee=500.0,
            annual_fee=500.0,
            welcome_offer="500 cashback on first transaction within 30 days",
            primary_category=SpendingCategory.ONLINE,
            categories=[SpendingCategory.ONLINE, SpendingCategory.GROCERIES, SpendingCategory.GENERAL],
            benefits=[
                "2× cashback on online spends",
                "1× cashback on offline spends",
                "1% fuel surcharge waiver",
                "EMI conversion facility",
            ],
            reward_rate=2.0,
            annual_benefit_value=5000.0,
            min_income_required=300_000.0,
            credit_score_required=CreditScoreTier.GOOD,
        ),
        CreditCard(
            card_name="SBI SimplyCLICK",
            bank_name="State Bank of India",
            joining_fee=499.0,
            annual_fee=499.0,
            welcome_offer="Amazon gift card worth INR 500 on joining",
            primary_category=SpendingCategory.ONLINE,
            categories=[SpendingCategory.ONLINE, SpendingCategory.DINING, SpendingCategory.ENTERTAINMENT],
            benefits=[
                "10× reward points on Amazon, Cleartrip, Lenskart",
                "5× reward points on all other online spends",
                "1× reward points offline",
                "Annual fee waiver on INR 1L annual spend",
            ],
            reward_rate=5.0,
            annual_benefit_value=8000.0,
            min_income_required=300_000.0,
            credit_score_required=CreditScoreTier.GOOD,
        ),
        CreditCard(
            card_name="Axis Bank Ace",
            bank_name="Axis Bank",
            joining_fee=499.0,
            annual_fee=499.0,
            welcome_offer="INR 500 cashback on first transaction",
            primary_category=SpendingCategory.ONLINE,
            categories=[SpendingCategory.ONLINE, SpendingCategory.DINING, SpendingCategory.FUEL, SpendingCategory.GROCERIES],
            benefits=[
                "5% cashback on bill payments via Google Pay",
                "4% cashback on Swiggy, Zomato, Ola",
                "2% cashback on all other spends",
                "1% fuel surcharge waiver",
                "Annual fee waiver on INR 2L spend",
            ],
            reward_rate=2.0,
            annual_benefit_value=9000.0,
            min_income_required=300_000.0,
            credit_score_required=CreditScoreTier.GOOD,
        ),
        CreditCard(
            card_name="ICICI Amazon Pay",
            bank_name="ICICI Bank",
            joining_fee=0.0,
            annual_fee=0.0,
            welcome_offer="INR 500 Amazon Pay cashback on activation",
            primary_category=SpendingCategory.ONLINE,
            categories=[SpendingCategory.ONLINE, SpendingCategory.FUEL, SpendingCategory.GENERAL],
            benefits=[
                "5% cashback on Amazon (Prime members)",
                "3% cashback on Amazon (non-Prime members)",
                "2% cashback on Amazon Pay partner merchants",
                "1% cashback on all other transactions",
                "No joining or annual fee",
            ],
            reward_rate=1.0,
            annual_benefit_value=6000.0,
            min_income_required=0.0,
            credit_score_required=CreditScoreTier.FAIR,
        ),
        CreditCard(
            card_name="ICICI Coral",
            bank_name="ICICI Bank",
            joining_fee=500.0,
            annual_fee=500.0,
            welcome_offer="Payback points worth INR 250",
            primary_category=SpendingCategory.DINING,
            categories=[SpendingCategory.DINING, SpendingCategory.ENTERTAINMENT, SpendingCategory.GROCERIES],
            benefits=[
                "2 Payback points per INR 100 on dining and groceries",
                "25% discount at partner restaurants",
                "Buy 1 Get 1 movie ticket on BookMyShow (2/month)",
                "1 complimentary airport lounge per quarter",
            ],
            reward_rate=2.0,
            annual_benefit_value=7000.0,
            min_income_required=350_000.0,
            credit_score_required=CreditScoreTier.GOOD,
        ),
        CreditCard(
            card_name="Citi Cashback",
            bank_name="Citi Bank",
            joining_fee=500.0,
            annual_fee=500.0,
            welcome_offer="INR 300 cashback on first utility bill",
            primary_category=SpendingCategory.GENERAL,
            categories=[SpendingCategory.GENERAL, SpendingCategory.DINING, SpendingCategory.GROCERIES],
            benefits=[
                "5% cashback on movie tickets, telecom, utility bills",
                "0.5% cashback on all other transactions",
                "Auto-credit cashback to statement",
            ],
            reward_rate=0.5,
            annual_benefit_value=4500.0,
            min_income_required=300_000.0,
            credit_score_required=CreditScoreTier.GOOD,
        ),
        CreditCard(
            card_name="BPCL SBI Card Octane",
            bank_name="State Bank of India",
            joining_fee=1499.0,
            annual_fee=1499.0,
            welcome_offer="6000 reward points on joining",
            primary_category=SpendingCategory.FUEL,
            categories=[SpendingCategory.FUEL, SpendingCategory.DINING, SpendingCategory.GROCERIES],
            benefits=[
                "7.25% value back on BPCL fuel",
                "Waiver of 1% fuel surcharge at BPCL pumps",
                "25× reward points on BPCL spends",
                "10× reward points on dining and groceries",
                "Annual fee waiver on INR 2L spend",
            ],
            reward_rate=7.25,
            annual_benefit_value=18000.0,
            min_income_required=500_000.0,
            credit_score_required=CreditScoreTier.GOOD,
        ),
        CreditCard(
            card_name="Axis Bank Vistara",
            bank_name="Axis Bank",
            joining_fee=1500.0,
            annual_fee=1500.0,
            welcome_offer="1 complimentary economy class ticket on activation",
            primary_category=SpendingCategory.TRAVEL,
            categories=[SpendingCategory.TRAVEL, SpendingCategory.DINING, SpendingCategory.ONLINE],
            benefits=[
                "Club Vistara points on every INR 200 spent",
                "1 complimentary economy ticket on INR 1.5L spend",
                "Complimentary lounge access (domestic)",
                "Travel insurance cover",
                "Priority check-in on Vistara flights",
            ],
            reward_rate=2.5,
            annual_benefit_value=15000.0,
            min_income_required=600_000.0,
            credit_score_required=CreditScoreTier.VERY_GOOD,
        ),
        CreditCard(
            card_name="Standard Chartered Super Value Titanium",
            bank_name="Standard Chartered",
            joining_fee=750.0,
            annual_fee=750.0,
            welcome_offer="INR 1000 cashback on first 3 transactions",
            primary_category=SpendingCategory.FUEL,
            categories=[SpendingCategory.FUEL, SpendingCategory.GENERAL],
            benefits=[
                "5% cashback on fuel, phone bills, and utility bills",
                "1 reward point per INR 150 on other spends",
                "Annual fee waiver on INR 90K annual spend",
            ],
            reward_rate=5.0,
            annual_benefit_value=7500.0,
            min_income_required=300_000.0,
            credit_score_required=CreditScoreTier.GOOD,
        ),
        CreditCard(
            card_name="Yes Bank PaisaSave",
            bank_name="Yes Bank",
            joining_fee=0.0,
            annual_fee=0.0,
            welcome_offer="INR 250 cashback on first transaction",
            primary_category=SpendingCategory.GROCERIES,
            categories=[SpendingCategory.GROCERIES, SpendingCategory.ONLINE, SpendingCategory.GENERAL],
            benefits=[
                "5% cashback on BigBazaar, D-Mart, and grocery stores",
                "1% cashback on all other purchases",
                "No annual fee ever",
            ],
            reward_rate=1.0,
            annual_benefit_value=4000.0,
            min_income_required=200_000.0,
            credit_score_required=CreditScoreTier.FAIR,
        ),
        CreditCard(
            card_name="HDFC Millennia",
            bank_name="HDFC Bank",
            joining_fee=1000.0,
            annual_fee=1000.0,
            welcome_offer="1000 CashPoints on joining",
            primary_category=SpendingCategory.ONLINE,
            categories=[SpendingCategory.ONLINE, SpendingCategory.DINING, SpendingCategory.GROCERIES, SpendingCategory.ENTERTAINMENT],
            benefits=[
                "5% cashback on Amazon, Flipkart, Myntra",
                "2.5% cashback on all online transactions",
                "1% cashback on offline and wallet transactions",
                "Quarterly milestone bonus cashback",
            ],
            reward_rate=2.5,
            annual_benefit_value=11000.0,
            min_income_required=350_000.0,
            credit_score_required=CreditScoreTier.GOOD,
        ),
        CreditCard(
            card_name="Kotak Mahindra PVR Platinum",
            bank_name="Kotak Mahindra Bank",
            joining_fee=499.0,
            annual_fee=499.0,
            welcome_offer="4 free PVR tickets on joining",
            primary_category=SpendingCategory.ENTERTAINMENT,
            categories=[SpendingCategory.ENTERTAINMENT, SpendingCategory.DINING, SpendingCategory.ONLINE],
            benefits=[
                "2 complimentary PVR tickets per month",
                "1 reward point per INR 100 on all spends",
                "Discount coupons for PVR F&B",
                "1% fuel surcharge waiver",
            ],
            reward_rate=1.0,
            annual_benefit_value=8000.0,
            min_income_required=300_000.0,
            credit_score_required=CreditScoreTier.GOOD,
        ),
        CreditCard(
            card_name="Amex Gold Charge",
            bank_name="American Express",
            joining_fee=1000.0,
            annual_fee=4500.0,
            welcome_offer="5000 Membership Rewards Points on INR 5K spend in 90 days",
            primary_category=SpendingCategory.DINING,
            categories=[SpendingCategory.DINING, SpendingCategory.TRAVEL, SpendingCategory.ONLINE, SpendingCategory.GROCERIES],
            benefits=[
                "1 Membership Reward point per INR 50 on all purchases",
                "3× points at partner restaurants",
                "Travel and medical insurance",
                "24/7 American Express customer service",
                "Points redeemable across airline miles and hotel rewards",
            ],
            reward_rate=2.0,
            annual_benefit_value=20000.0,
            min_income_required=600_000.0,
            credit_score_required=CreditScoreTier.VERY_GOOD,
        ),
        CreditCard(
            card_name="IndusInd Tiger",
            bank_name="IndusInd Bank",
            joining_fee=0.0,
            annual_fee=0.0,
            welcome_offer="Assured cashback on first 3 transactions",
            primary_category=SpendingCategory.FUEL,
            categories=[SpendingCategory.FUEL, SpendingCategory.GENERAL],
            benefits=[
                "1% cashback on all fuel transactions",
                "No joining or annual fee",
                "Contactless payment enabled",
            ],
            reward_rate=1.0,
            annual_benefit_value=2000.0,
            min_income_required=180_000.0,
            credit_score_required=CreditScoreTier.FAIR,
        ),
    ]


# ---------------------------------------------------------------------------
# Public loader entry point
# ---------------------------------------------------------------------------

def load_cards(excel_path: str | Path | None = None) -> List[CreditCard]:
    """
    Load credit cards from Excel if available, else fall back to synthetic data.

    Args:
        excel_path: Path to the .xlsx file. If None, tries the default location.

    Returns:
        List[CreditCard]
    """
    default_paths = [
        Path("data/credit_cards.xlsx"),
        Path("credit_cards.xlsx"),
    ]

    if excel_path is not None:
        paths_to_try = [Path(excel_path)]
    else:
        paths_to_try = default_paths

    for p in paths_to_try:
        if p.exists():
            try:
                cards = load_from_excel(p)
                if cards:
                    return cards
            except Exception as exc:
                logger.warning("Failed to load from %s: %s. Falling back to synthetic data.", p, exc)

    logger.info("No Excel file found or load failed — using synthetic card dataset.")
    return load_synthetic_cards()
