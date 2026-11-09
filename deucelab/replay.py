import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from deucelab.constants import STANDARD_OU_LINES
from deucelab.markets import valid_price
from deucelab.rules import is_finished, is_player

MAX_REPLAY_LINE = 1_048_576
REPLAY_METADATA = ("match_id", "set_number", "quote_at", "settled_at")


def validate_rallies(rallies: tuple[int, ...]) -> None:
    p1 = p2 = 0
    for winner in rallies:
        if not is_player(winner):
            raise ValueError("rally winners must be 1 or 2")
        if is_finished(p1, p2):
            raise ValueError("rallies continue after the set ended")
        p1 += winner == 1
        p2 += winner == 2
    if not is_finished(p1, p2):
        raise ValueError("incomplete set")


@dataclass(frozen=True)
class SetReplay:
    first_server: int
    rallies: tuple[int, ...]
    line: float
    over: float
    under: float
    prematch: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not is_player(self.first_server):
            raise ValueError("first_server must be 1 or 2")
        if self.line not in STANDARD_OU_LINES:
            raise ValueError("unsupported O/U line")
        if not isinstance(self.rallies, (list, tuple)):
            raise ValueError("rallies must be a sequence of player numbers")
        object.__setattr__(self, "rallies", tuple(self.rallies))
        prices: tuple[float, ...] = (self.over, self.under)
        if self.prematch is not None:
            if not isinstance(self.prematch, (list, tuple)) or len(self.prematch) != 2:
                raise ValueError("prematch must contain two prices")
            object.__setattr__(self, "prematch", tuple(self.prematch))
            prices += tuple(self.prematch)
        if not all(valid_price(price) for price in prices):
            raise ValueError("prices must be finite decimal odds greater than 1")
        validate_rallies(self.rallies)


class ReplayMetadata(NamedTuple):
    match_id: str
    set_number: int
    quote_at: datetime
    settled_at: datetime


def parse_metadata(metadata: dict[str, object]) -> ReplayMetadata:
    match = metadata["match_id"]
    set_number = metadata["set_number"]
    quote_at, settled_at = metadata["quote_at"], metadata["settled_at"]
    if not isinstance(match, str) or not match.strip() or match != match.strip():
        raise ValueError("match_id must be a nonempty string without surrounding whitespace")
    if type(set_number) is not int or set_number < 1:
        raise ValueError("set_number must be a positive integer")
    if not isinstance(quote_at, str) or not isinstance(settled_at, str):
        raise ValueError("replay timestamps must use ISO 8601")
    quote = datetime.fromisoformat(quote_at)
    settlement = datetime.fromisoformat(settled_at)
    if quote.utcoffset() is None or settlement.utcoffset() is None or quote >= settlement:
        raise ValueError(
            "quote_at and settled_at need time zones and quote must precede settlement"
        )
    return ReplayMetadata(match, set_number, quote, settlement)


def parse_row(row: object, after_points: int) -> tuple[SetReplay, ReplayMetadata | None]:
    if not isinstance(row, dict):
        raise ValueError("each replay row must be an object")
    data = row.copy()
    checkpoint = data.pop("after_points", None)
    if type(checkpoint) is not int or checkpoint != after_points or after_points < 0:
        raise ValueError("quote snapshot does not match --after-points")
    metadata_fields = {key: data.pop(key) for key in REPLAY_METADATA if key in data}
    if metadata_fields and len(metadata_fields) != len(REPLAY_METADATA):
        raise ValueError("provide all match identity and timestamp fields together")
    metadata = parse_metadata(metadata_fields) if metadata_fields else None
    replay = SetReplay(**data)
    if metadata is not None and after_points >= len(replay.rallies):
        raise ValueError("quote snapshot must precede the end of the set")
    return replay, metadata


def _unique_fields(fields: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(fields)
    if len(result) != len(fields):
        raise ValueError("duplicate JSON keys")
    return result


def read_replay_rows(
    path: Path,
    after_points: int,
) -> Iterator[tuple[SetReplay, ReplayMetadata | None]]:
    seen: set[tuple[str, int] | SetReplay] = set()
    match_times: dict[str, list[ReplayMetadata]] = defaultdict(list)
    with path.open(encoding="utf-8-sig") as source:
        for number, line in enumerate(
            iter(lambda: source.readline(MAX_REPLAY_LINE + 1), ""), start=1
        ):
            if len(line) > MAX_REPLAY_LINE:
                raise ValueError(f"{path}, line {number}: replay exceeds the row size limit")
            if not line.strip():
                continue
            try:
                row = json.loads(line, object_pairs_hook=_unique_fields)
                replay, metadata = parse_row(row, after_points)
                identity = (metadata.match_id, metadata.set_number) if metadata else replay
                if identity in seen:
                    raise ValueError("duplicate replay")
                seen.add(identity)
                if metadata is not None:
                    match_times[metadata.match_id].append(metadata)
                yield replay, metadata
            except (RecursionError, TypeError, ValueError) as error:
                raise ValueError(f"{path}, line {number}: {error}") from error

    for match_id, times in match_times.items():
        ordered = sorted(times, key=lambda row: row.set_number)
        if any(a.settled_at > b.quote_at for a, b in zip(ordered, ordered[1:])):
            raise ValueError(f"{path}: inconsistent set chronology for match {match_id!r}")


def read_replays(path: Path, after_points: int) -> Iterator[SetReplay]:
    for replay, _ in read_replay_rows(path, after_points):
        yield replay


def encode_replay(payload: dict[str, object], after_points: int) -> str:
    _, metadata = parse_row(payload, after_points)
    if metadata is None:
        raise ValueError("an export needs match identity and timestamps")
    result = json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n"
    if len(result) > MAX_REPLAY_LINE:
        raise ValueError("replay exceeds the row size limit")
    return result


def write_replay(path: Path, row: str) -> None:
    stream = path.open("x", encoding="utf-8", newline="\n")
    try:
        with stream:
            stream.write(row)
    except OSError:
        path.unlink(missing_ok=True)
        raise
