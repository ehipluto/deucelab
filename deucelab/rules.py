Score = tuple[int, int]


def is_player(player: int) -> bool:
    return type(player) is int and player in (1, 2)


def is_score(p1: int, p2: int) -> bool:
    if type(p1) is not int or type(p2) is not int or min(p1, p2) < 0:
        return False
    winner, loser = max(p1, p2), min(p1, p2)
    return winner <= max(11, loser + 2)


def is_finished(p1: int, p2: int) -> bool:
    return max(p1, p2) >= 11 and abs(p1 - p2) >= 2


def next_server(p1: int, p2: int, first_server: int) -> int:
    if not is_score(p1, p2) or not is_player(first_server):
        raise ValueError("expected a reachable score and a first server of 1 or 2")
    turns = p1 + p2 - 20 if min(p1, p2) >= 10 else (p1 + p2) // 2
    return 3 - first_server if turns % 2 else first_server
