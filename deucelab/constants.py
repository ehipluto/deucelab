BASE_SERVER_WIN_PROB: float = 0.61
SKILL_DIFF_POINT_IMPACT: float = 0.18
SERVER_ADV_SKILL_MODIFIER: float = 0.5

CACHE_SIZE_SIMULATION: int = 4096

CACHE_SIZE_POINT_PROB: int = 256

MOMENTUM_WINDOW: int = 10
MOMENTUM_DECAY_FACTOR: float = 0.85
MOMENTUM_IMPACT_SCALE: float = 0.06


PERFORMANCE_ADJUSTMENT_FACTOR: float = 0.08
HISTORY_WEIGHTING_RECENCY: int = 3
W_MOMENTUM = 0.30
W_PERF_VS_ODDS = 0.35
W_PRESSURE = 0.15
W_SERVE_PERF = 0.20

OU_LINE_SERVER_FACTOR: float = 0.15

ADAPTIVE_ELAB_THRESHOLD_ENABLED: bool = True
ADAPTIVE_ELAB_BASE_POINTS: int = 7
ADAPTIVE_ELAB_MIN_POINTS: int = 6
ADAPTIVE_ELAB_MAX_POINTS: int = 15
ADAPTIVE_ELAB_HISTORY_LEN: int = 10
ADAPTIVE_ELAB_VOLATILITY_SCALE: float = 50.0

STANDARD_OU_LINES: list[float] = [14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5]
DEFAULT_OU_EVALUATION_RANGE: float = 1.0

DEFAULT_MIN_PROBABILITY_BASE: float = 0.60
DEFAULT_MAX_PROB_FILTER_BASE: float = 0.95
DEFAULT_MIN_ESTIMATED_ODDS_BASE: float = 1.55
DEFAULT_MIN_ACCEPTABLE_EV_BASE: float = 0.07

DEFAULT_CONFIG = {
    "min_acceptable_ev_base": DEFAULT_MIN_ACCEPTABLE_EV_BASE,
    "min_estimated_odds_base": DEFAULT_MIN_ESTIMATED_ODDS_BASE,
    "min_prob_suggestion_base": DEFAULT_MIN_PROBABILITY_BASE,
    "max_prob_filter_base": DEFAULT_MAX_PROB_FILTER_BASE,
    "ou_evaluation_range": DEFAULT_OU_EVALUATION_RANGE,
    "adaptive_elab_threshold_enabled": ADAPTIVE_ELAB_THRESHOLD_ENABLED,
    "adaptive_elab_base_points": ADAPTIVE_ELAB_BASE_POINTS,
    "adaptive_elab_volatility_scale": ADAPTIVE_ELAB_VOLATILITY_SCALE,
    "ou_line_server_factor": OU_LINE_SERVER_FACTOR,
    "momentum_impact_scale": MOMENTUM_IMPACT_SCALE,
    "momentum_decay_factor": MOMENTUM_DECAY_FACTOR,
}

THEME_COLOR = "blue"
APPEARANCE_MODE = "System"

TOOLTIPS = {
    "min_prob_suggestion_base": (
        "Probabilità minima calcolata (0-1) affinché un suggerimento venga considerato."
    ),
    "max_prob_filter_base": (
        "Probabilità massima calcolata (0-1) oltre la quale un suggerimento "
        "viene scartato (troppo ovvio)."
    ),
    "min_estimated_odds_base": "Quota minima stimata (>1.0) richiesta per un suggerimento.",
    "min_acceptable_ev_base": (
        "Valore Atteso (EV) minimo (0-1) richiesto per un suggerimento (es. 0.07 = 7%)."
    ),
    "ou_evaluation_range": (
        "Intervallo di linee O/U da valutare attorno alla linea centrale predetta "
        "per le scommesse 'Elaborated' (es: 1.0 -> valuta Centrale +/- 1.0)."
    ),
    "adaptive_elab_threshold_enabled": (
        "Abilita/Disabilita la soglia adattiva per le scommesse elaborate basata sulla volatilità."
    ),
    "adaptive_elab_base_points": (
        "Punti base necessari per l'elaborazione prima di considerare la volatilità."
    ),
    "adaptive_elab_volatility_scale": (
        "Fattore di scala: quanto la volatilità aumenta i punti necessari per l'elaborazione."
    ),
    "ou_line_server_factor": (
        "Quanto la probabilità del server corrente influenza la linea O/U suggerita (0=nessuna influenza)."
    ),
    "momentum_impact_scale": (
        "Impatto massimo (boost/penalty 0-1) del momentum esponenziale sulla probabilità punto."
    ),
    "momentum_decay_factor": (
        "Fattore di decadimento (0-1) per il momentum esponenziale (valori più alti = decadimento più lento)."
    ),
}


def valid_config(values: dict[str, object]) -> bool:
    from math import isfinite

    if type(values["adaptive_elab_threshold_enabled"]) is not bool:
        return False
    points = values["adaptive_elab_base_points"]
    if type(points) is not int or not 1 <= points <= ADAPTIVE_ELAB_MAX_POINTS:
        return False
    for key, value in values.items():
        if key in ("adaptive_elab_threshold_enabled", "adaptive_elab_base_points"):
            continue
        if type(value) not in (int, float) or abs(value) > float.fromhex("0x1.fffffffffffffp1023"):
            return False
        if not isfinite(value) or value < 0:
            return False
        if key == "min_estimated_odds_base" and value <= 1:
            return False
        if (
            key
            in (
                "min_prob_suggestion_base",
                "max_prob_filter_base",
                "momentum_impact_scale",
                "momentum_decay_factor",
            )
            and value > 1
        ):
            return False
    return values["min_prob_suggestion_base"] <= values["max_prob_filter_base"]
