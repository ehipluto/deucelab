import math
from collections.abc import Callable

from deucelab.constants import STANDARD_OU_LINES
from deucelab.rules import is_finished, next_server

CAP = math.ceil(max(STANDARD_OU_LINES))
TERMINAL = tuple(
    tuple(float(index == total) for index in range(CAP + 1)) for total in range(CAP + 1)
)


def totals_step(
    p1: int,
    p2: int,
    first_server: int,
    probability_for_server: Callable[[int], float],
    recurse: Callable[[int, int], tuple[float, ...]],
) -> tuple[float, ...]:
    total = p1 + p2
    if total >= CAP or is_finished(p1, p2):
        return TERMINAL[min(total, CAP)]

    probability = probability_for_server(next_server(p1, p2, first_server))
    win = recurse(p1 + 1, p2)
    loss = recurse(p1, p2 + 1)
    return tuple(probability * a + (1 - probability) * b for a, b in zip(win, loss))
