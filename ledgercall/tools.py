"""Deterministic pure-Python finance tools for LedgerCall.

All tool functions use strict type annotations and Google-style docstrings so
AgentTune (`transformers.utils.get_json_schema` / fallback schema builder) can
extract clean JSON schemas automatically.
"""

from __future__ import annotations

from collections.abc import Callable

# Fixed deterministic exchange-rate table (units of currency per 1.00 USD).
FX_RATES_PER_USD: dict[str, float] = {
    "USD": 1.00,
    "EUR": 0.92,
    "GBP": 0.79,
    "JPY": 152.00,
    "INR": 84.00,
}

SUPPORTED_CURRENCIES: tuple[str, ...] = tuple(FX_RATES_PER_USD.keys())


def convert_currency(amount: float, from_ccy: str, to_ccy: str) -> float:
    """Convert a monetary amount between supported currencies using a fixed FX rate table.

    Supported currency codes are USD, EUR, GBP, JPY, and INR.

    Args:
        amount: Monetary amount to convert in the source currency.
        from_ccy: Three-letter ISO code of the source currency (USD, EUR, GBP, JPY, or INR).
        to_ccy: Three-letter ISO code of the target currency (USD, EUR, GBP, JPY, or INR).

    Returns:
        Converted monetary amount in the target currency, rounded to 2 decimal places.
    """
    amt = float(amount)
    src = str(from_ccy).strip().upper()
    dst = str(to_ccy).strip().upper()
    if src not in FX_RATES_PER_USD:
        raise ValueError(f"Unsupported currency '{from_ccy}'. Supported: {', '.join(SUPPORTED_CURRENCIES)}")
    if dst not in FX_RATES_PER_USD:
        raise ValueError(f"Unsupported currency '{to_ccy}'. Supported: {', '.join(SUPPORTED_CURRENCIES)}")
    if src == dst:
        return round(amt, 2)
    usd_value = amt / FX_RATES_PER_USD[src]
    return round(usd_value * FX_RATES_PER_USD[dst], 2)


def compound_interest(
    principal: float,
    annual_rate_pct: float,
    years: int,
    compounds_per_year: int,
) -> float:
    """Compute the final total balance (principal plus interest) under compound interest.

    Uses the standard formula A = P * (1 + r / n) ** (n * t), where r is
    annual_rate_pct / 100, n is compounds_per_year, and t is years.

    Args:
        principal: Initial principal amount invested or deposited.
        annual_rate_pct: Nominal annual interest rate as a percentage (e.g. 5.5 for 5.5%).
        years: Number of whole years the principal is compounded.
        compounds_per_year: Number of compounding periods per year (e.g. 1, 2, 4, or 12).

    Returns:
        Final maturity balance including principal and accrued interest, rounded to 2 decimal places.
    """
    p = float(principal)
    r = float(annual_rate_pct) / 100.0
    t = int(years)
    n = int(compounds_per_year)
    if t < 0:
        raise ValueError("years must be non-negative")
    if n <= 0:
        raise ValueError("compounds_per_year must be a positive integer")
    final_balance = p * ((1.0 + r / n) ** (n * t))
    return round(final_balance, 2)


def percent_change(old: float, new: float) -> float:
    """Calculate the percentage change from an old value to a new value.

    Uses ((new - old) / old) * 100. For example, old=100 and new=125 returns 25.0.

    Args:
        old: Original baseline value (must be non-zero).
        new: New updated value to compare against the baseline.

    Returns:
        Signed percentage change from old to new, rounded to 2 decimal places.
    """
    old_val = float(old)
    new_val = float(new)
    if old_val == 0.0:
        raise ValueError("old value must be non-zero")
    return round(((new_val - old_val) / old_val) * 100.0, 2)


def loan_payment(principal: float, annual_rate_pct: float, months: int) -> float:
    """Compute the fixed monthly payment for a fully amortizing loan.

    Uses M = P * (i * (1 + i)**m) / ((1 + i)**m - 1) where i is the monthly
    interest rate (annual_rate_pct / 100 / 12) and m is the loan term in months.

    Args:
        principal: Total loan principal borrowed.
        annual_rate_pct: Annual percentage interest rate (e.g. 6.5 for 6.5% APR).
        months: Total repayment period in months (must be positive).

    Returns:
        Monthly loan payment amount, rounded to 2 decimal places.
    """
    p = float(principal)
    apr = float(annual_rate_pct)
    m = int(months)
    if m <= 0:
        raise ValueError("months must be a positive integer")
    if apr < 0:
        raise ValueError("annual_rate_pct must be non-negative")
    monthly_rate = (apr / 100.0) / 12.0
    if monthly_rate == 0.0:
        return round(p / m, 2)
    factor = (1.0 + monthly_rate) ** m
    payment = p * (monthly_rate * factor) / (factor - 1.0)
    return round(payment, 2)


TOOLS: list[Callable[..., float]] = [
    convert_currency,
    compound_interest,
    percent_change,
    loan_payment,
]

TOOL_NAMES: list[str] = [fn.__name__ for fn in TOOLS]

TOOL_MAP: dict[str, Callable[..., float]] = {fn.__name__: fn for fn in TOOLS}
