"""Fog of war: nasconde l'Opponent oltre il raggio di vista, solo nell'osservazione.

Non modifica GameState, regole o maschere e non estrae numeri casuali.
Un piano Opponent tutto a zero significa "non visibile" (l'Opponent esiste sempre).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .game import GameState
from .observations import SpatialChannel, _array_indices
from .rules import SHOOT_MAX_DISTANCE, manhattan_distance


def validate_fog_radius(radius: int) -> None:
    if type(radius) is not int or radius < SHOOT_MAX_DISTANCE:
        raise ValueError(
            f"fog_radius deve essere un intero >= {SHOOT_MAX_DISTANCE} "
            "(portata di tiro): altrimenti la maschera rivela l'Opponent"
        )


def apply_fog_of_war(
    spatial: np.ndarray, states: Sequence[GameState], radius: int
) -> np.ndarray:
    """Copia di (N,4,9,9) con il piano Opponent azzerato se distanza > radius."""
    validate_fog_radius(radius)
    if spatial.ndim != 4 or spatial.shape[0] != len(states):
        raise ValueError("spatial e states devono avere lo stesso batch size")
    out = spatial.copy()
    for i, state in enumerate(states):
        if manhattan_distance(state.agent_position, state.opponent_position) > radius:
            out[i, SpatialChannel.OPPONENT] = 0.0
    return out


__all__ = ["apply_fog_of_war", "validate_fog_radius"]
