"""Bayesian filtering sull'identità nascosta dell'Opponent (POMDP).

La mappa e lo stato dinamico X_t sono osservabili; la policy K e' campionata
una sola volta per episodio e non cambia durante le 100 attivazioni.

    b_t[k] = P(K = k | storia osservata, mappa)

Quando l'Opponent sceglie un'azione osservabile u_t nello stato PRE-azione
X_t, il posterior e':

    b_{t+1}[k] = b_t[k] * pi_k(u_t | X_t, m) /
                   sum_j b_t[j] * pi_j(u_t | X_t, m)

Le likelihood pi_k sono calcolate dalle sei policy ESATTE gia' definite in
``environment.opponents``. Non si utilizzano mai l'identita' reale della
policy o gli esiti del combattimento per aggiornare il belief.

Nelle attivazioni dell'Agent e nei melee obbligatori il belief resta invariato.
Il respawn NON lo azzera; soltanto l'inizio di un nuovo episodio lo riporta
al prior (di default uniforme anche durante la valutazione contro un singolo
opponent fissato dal simulatore).

Ordine dei sei elementi, identico a OPPONENT_POOL:
    Random, Aggressive, Hunter, Hybrid, Shooter, ASH.

I calcoli utilizzano float64; per la rete neurale si puo' convertire il
vettore di probabilita' a float32 al momento della codifica dell'osservazione.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Sequence

import numpy as np

from .maps import GameMap
from .opponents import (
    OPPONENT_POOL,
    OpponentKind,
    OpponentState,
    opponent_action_probabilities,
)
from .rules import Action, Player, are_adjacent

if TYPE_CHECKING:
    from .game import GameState, StepResult


OPPONENT_ORDER: Final[tuple[OpponentKind, ...]] = OPPONENT_POOL
N_OPPONENTS: Final[int] = len(OPPONENT_ORDER)
_UNIFORM_PRIOR: Final[tuple[float, ...]] = (1.0 / N_OPPONENTS,) * N_OPPONENTS
_NORMALIZATION_TOLERANCE: Final[float] = 1e-10


def validate_belief(
    probabilities: Sequence[float] | np.ndarray, *, name: str = "belief"
) -> np.ndarray:
    """Valida e restituisce una COPIA normalizzata di una distribuzione.

    Gli elementi devono essere sei numeri finiti e non negativi, con somma
    uguale a uno entro la tolleranza degli errori floating-point. Non si
    correggono silenziosamente distribuzioni significativamente non valide.
    """
    try:
        raw = np.asarray(probabilities)
        if raw.dtype.kind == "b":
            raise ValueError(f"{name} non puo' contenere valori booleani")
        vector = np.asarray(probabilities, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} deve contenere {N_OPPONENTS} numeri reali") from exc

    if vector.shape != (N_OPPONENTS,):
        raise ValueError(
            f"{name} deve avere forma ({N_OPPONENTS},), non {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contiene probabilita' non finite")
    if np.any(vector < 0.0):
        raise ValueError(f"{name} contiene probabilita' negative")

    total = float(np.sum(vector, dtype=np.float64))
    if not np.isclose(total, 1.0, rtol=0.0, atol=_NORMALIZATION_TOLERANCE):
        raise ValueError(f"Le probabilita' di {name} devono sommare a uno")

    # Copia indipendente dall'input, anche quando e' gia' un ndarray.
    return np.array(vector / total, dtype=np.float64, copy=True)


def initial_belief(
    prior: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Crea b_1 = q_O come vettore float64 nell'ordine OPPONENT_ORDER.

    Il prior default e' uniforme. Un prior esplicito va passato solo se e'
    l'informazione realmente disponibile all'Agent; NON utilizzare una
    distribuzione delta creata dal valutatore per fissare il vero opponent.
    """
    return validate_belief(_UNIFORM_PRIOR if prior is None else prior, name="prior")


def _as_opponent_state(state: OpponentState | GameState) -> OpponentState:
    """Accetta sia la vista OpponentState sia GameState del simulatore.

    La conversione usa esclusivamente la parte osservabile dello stato;
    non accede al contesto, alla vera policy o agli RNG dell'ambiente.
    """
    if isinstance(state, OpponentState):
        return state
    try:
        return OpponentState(
            agent_position=state.agent_position,
            opponent_position=state.opponent_position,
            active_objectives=state.active_objectives,
            score_differential=state.score_differential,
            player_to_act=state.player_to_act,
            activation=state.activation,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(
            "state deve essere un GameState o un OpponentState valido"
        ) from exc


def _as_primitive_action(action: Action | int) -> Action:
    """Converte l'azione osservata; DUMMY non e' una scelta dell'Opponent."""
    if isinstance(action, Action):
        result = action
    elif type(action) is int:
        try:
            result = Action(action)
        except ValueError as exc:
            raise ValueError(f"Azione osservata sconosciuta: {action!r}") from exc
    else:
        raise ValueError(f"Azione osservata non valida: {action!r}")

    if result is Action.DUMMY:
        raise ValueError("DUMMY non e' un'azione osservabile scelta dall'Opponent")
    return result


def action_likelihoods(
    opponent_state: OpponentState | GameState,
    game_map: GameMap,
    observed_action: Action | int,
) -> np.ndarray:
    """Calcola [pi_k(u | X_t,m)]_{k=1..6}, senza fare alcuna estrazione.

    Richiede lo stato PRIMA della scelta dell'Opponent, durante una sua
    attivazione non forzata. Un'azione impossibile riceve likelihood zero.
    """
    state = _as_opponent_state(opponent_state)
    action = _as_primitive_action(observed_action)
    if state.player_to_act is not Player.OPPONENT:
        raise ValueError("Il belief si aggiorna solo dopo decisioni dell'Opponent")
    if are_adjacent(state.agent_position, state.opponent_position):
        raise ValueError("Nel melee obbligatorio non esiste azione da inferire")

    likelihoods = np.fromiter(
        (
            opponent_action_probabilities(kind, state, game_map).get(action, 0.0)
            for kind in OPPONENT_ORDER
        ),
        dtype=np.float64,
        count=N_OPPONENTS,
    )
    if not np.all(np.isfinite(likelihoods)) or np.any(likelihoods < 0.0):
        raise RuntimeError("Le policy hanno restituito likelihood non valide")
    return likelihoods


def update_belief(
    prior: Sequence[float] | np.ndarray,
    opponent_state: OpponentState | GameState,
    game_map: GameMap,
    observed_action: Action | int,
) -> np.ndarray:
    """Esegue un aggiornamento bayesiano esatto e restituisce un nuovo array.

    L'input non viene modificato. La normalizzazione richiede evidenza
    strettamente positiva. Un evento con evidenza nulla segnala una violazione
    delle ipotesi del modello (o del prior) e produce un errore esplicito:
    non e' corretto sostituire arbitrariamente il posterior col prior.
    """
    belief = validate_belief(prior)
    likelihoods = action_likelihoods(opponent_state, game_map, observed_action)
    unnormalized = belief * likelihoods
    evidence = float(np.sum(unnormalized, dtype=np.float64))

    if not np.isfinite(evidence) or evidence <= 0.0:
        raise ValueError(
            "Osservazione incompatibile con tutte le policy aventi "
            "probabilita' positiva nel prior (evidenza bayesiana nulla)"
        )

    posterior = unnormalized / evidence
    return validate_belief(posterior, name="posterior")


class BayesianOpponentBelief:
    """Memoria bayesiana ricorsiva per un SOLO episodio/ambiente.

    Ciascun worker parallelo deve possedere una propria istanza.

    Esempio::

        memory = BayesianOpponentBelief()
        memory.reset()  # inizio episodio, prior uniforme
        memory.observe_step(step_result, game_map)
        belief_features = memory.probabilities.astype(np.float32)
    """

    def __init__(
        self, prior: Sequence[float] | np.ndarray | None = None
    ) -> None:
        self._prior = initial_belief(prior)
        self._belief = self._prior.copy()

    @property
    def probabilities(self) -> np.ndarray:
        """Posterior attuale, restituito come copia indipendente."""
        return self._belief.copy()

    @property
    def prior(self) -> np.ndarray:
        """Prior applicato al reset dell'episodio, restituito come copia."""
        return self._prior.copy()

    def reset(
        self, prior: Sequence[float] | np.ndarray | None = None
    ) -> np.ndarray:
        """Reinizializza il belief SOLO quando comincia un nuovo episodio.

        Se prior non e' fornito, riutilizza il prior configurato in __init__
        (o nel reset precedente che lo abbia esplicitamente modificato).
        """
        if prior is not None:
            self._prior = initial_belief(prior)
        self._belief = self._prior.copy()
        return self.probabilities

    def update(
        self,
        opponent_state: OpponentState | GameState,
        game_map: GameMap,
        observed_action: Action | int,
    ) -> np.ndarray:
        """Aggiorna il posterior dopo una scelta osservabile dell'Opponent."""
        updated = update_belief(
            self._belief, opponent_state, game_map, observed_action
        )
        self._belief = updated
        return self.probabilities

    def observe_step(self, result: StepResult, game_map: GameMap) -> np.ndarray:
        """Consuma un StepResult senza mai usare l'identita' reale di K.

        Usa result.state (PRE-transizione) e result.events.effective_action.
        Le attivazioni Agent e i melee obbligatori non producono informazione
        sulle scelte della policy avversaria. L'eventuale morte, il respawn e
        la terminazione NON implicano un reset automatico del posterior.
        """
        previous_state = result.state
        if previous_state.player_to_act is not Player.OPPONENT:
            return self.probabilities
        if are_adjacent(
            previous_state.agent_position, previous_state.opponent_position
        ):
            return self.probabilities
        action = result.events.effective_action
        if action is None:
            raise ValueError(
                "Manca l'azione osservata dell'Opponent in un turno decisionale"
            )
        return self.update(previous_state, game_map, action)


__all__ = [
    "OPPONENT_ORDER",
    "N_OPPONENTS",
    "validate_belief",
    "initial_belief",
    "action_likelihoods",
    "update_belief",
    "BayesianOpponentBelief",
]
