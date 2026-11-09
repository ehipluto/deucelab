import math
from dataclasses import dataclass
from typing import Literal

from deucelab.rules import Score

Side = Literal["Over", "Under"]


def valid_price(price: object) -> bool:
    return (
        isinstance(price, (int, float))
        and not isinstance(price, bool)
        and 1 < price <= float.fromhex("0x1.fffffffffffffp1023")
    )


def no_margin(first: float, second: float) -> tuple[float, float]:
    if not valid_price(first) or not valid_price(second):
        raise ValueError("prices must be finite decimal odds greater than 1")
    first_implied, second_implied = 1 / first, 1 / second
    total = first_implied + second_implied
    return first_implied / total, second_implied / total


def expected_value(probability: float, price: float) -> float:
    if not 0 <= probability <= 1 or not valid_price(price):
        raise ValueError("expected a probability in [0, 1] and valid decimal odds")
    return probability * price - 1


def fair_odds(probability: float) -> float | None:
    if not 0 <= probability <= 1:
        raise ValueError("expected a probability in [0, 1]")
    if probability == 0:
        return None
    price = 1 / probability
    return price if math.isfinite(price) else None


@dataclass(frozen=True)
class Quote:
    id: int
    score: Score
    line: float | None
    first: float | None
    second: float | None

    @property
    def market_probability(self) -> tuple[float, float] | None:
        if self.first is None or self.second is None:
            return None
        return no_margin(self.first, self.second)
