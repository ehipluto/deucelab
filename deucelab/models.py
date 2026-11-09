from deucelab import constants as c

SERVE_EDGE = c.BASE_SERVER_WIN_PROB - 0.5


def direct_point_probability(
    strength: float,
    server: int,
    serve_edge: float = SERVE_EDGE,
) -> float:
    adjustment = serve_edge if server == 1 else -serve_edge
    return min(0.97, max(0.03, strength + adjustment))


def exponential_momentum(history: tuple[int, ...], decay: float, scale: float) -> float:
    recent = history[-c.MOMENTUM_WINDOW :]
    if len(recent) < 2:
        return 0.0
    momentum = 0.0
    total_weight = 0.0
    for index, winner in enumerate(reversed(recent)):
        weight = decay**index
        momentum += weight * (1.0 if winner == 1 else -1.0)
        total_weight += weight
    return min(1.0, max(-1.0, momentum / total_weight)) * scale


def legacy_point_probabilities(
    neutral_strength: float,
    server: int,
    momentum: float,
) -> tuple[float, float]:
    skill = neutral_strength - 0.5
    server_skill = skill if server == 1 else -skill
    impact = server_skill * (c.SERVER_ADV_SKILL_MODIFIER / 0.5) * abs(SERVE_EDGE)
    base = max(0.15, min(0.85, c.BASE_SERVER_WIN_PROB + impact))
    adjustment = momentum if server == 1 else -momentum
    server_probability = max(0.03, min(0.97, base + adjustment))
    receive_probability = 1.0 - server_probability
    if server == 1:
        return server_probability, receive_probability
    return receive_probability, server_probability
