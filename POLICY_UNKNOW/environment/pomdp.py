"""Wrapper POMDP per il simulatore originale ``environment.game.GridGame``.

L'unica componente nascosta dello stato completo e' l'identita' K della policy
avversaria, campionata una volta per episodio. L'Agent osserva:

    X_t = (pA, pO, G, D, tau, t), la mappa m e il belief b_t su K.

Osserva inoltre l'azione scelta dall'Opponent nella precedente attivazione,
se ce n'e' stata una. Il modello delle sei policy e' noto, ma la policy vera
non e' trasmessa dalle osservazioni pubbliche.

Questo modulo NON altera alcuna regola, probabilita', ricompensa o generatore
casuale di GridGame. La memoria bayesiana e' implementata da
``environment.belief.BayesianOpponentBelief``.

Esempio d'uso::

    from environment.pomdp import POMDPGame
    from environment.rules import Action

    game = POMDPGame(seed=42)
    observation = game.reset()  # prior uniforme sulle sei policy
    while not game.terminated:
        action = game.legal_actions()[0]  # solo esempio, non policy ottimale
        transition = game.step(action)
        if not transition.terminated:
            observation = game.observe()  # X_{t+1}, belief aggiornato

In evaluation, il simulatore puo' ricevere ``reset(context=contesto_fissato)``;
il belief dell'Agent parte comunque dal prior configurato (uniforme per
impostazione predefinita), NON dalla distribuzione delta del valutatore.

Nota: l'integrazione nel training parallelo e la codifica CNN/MLP richiedono
anche l'adeguamento di ``parallel_env.py`` e ``observations.py``. In
particolare, NON utilizzare ``encode_observation(state, game.context)``
nel percorso POMDP: quello e' il codificatore del vecchio MDP e inserisce
l'identita' vera della policy nelle feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np

from .belief import BayesianOpponentBelief, N_OPPONENTS
from .context import (
    DEFAULT_CONTEXT_DISTRIBUTION,
    ContextDistribution,
    EpisodeContext,
)
from .game import (
    GridGame,
    GameOutcome,
    GameState,
    StepResult,
    TERMINAL_STATE,
    TerminalState,
)
from .maps import GameMap
from .rules import Action, Player, are_adjacent


@dataclass(frozen=True, slots=True)
class POMDPObservation:
    """Informazione pubblica disponibile all'Agent prima dell'attivazione.

    ``state`` e' il solo stato dinamico osservabile X_t (GameState), non lo
    stato completo (X_t, K). ``game_map`` e' noto. ``belief`` e' una tupla
    immutabile di sei probabilita' nell'ordine OPPONENT_POOL.

    ``observed_opponent_action`` e' l'azione scelta dall'Opponent
    nell'attivazione immediatamente precedente, oppure None quando quella
    attivazione era dell'Agent o consisteva in un melee obbligatorio.
    None NON significa Pass: Pass e' il valore Action.PASS.
    """

    state: GameState
    game_map: GameMap
    belief: tuple[float, ...]
    observed_opponent_action: Action | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, GameState):
            raise TypeError("state deve essere un GameState non terminale")
        if not isinstance(self.game_map, GameMap):
            raise TypeError("game_map deve essere una GameMap")
        if len(self.belief) != N_OPPONENTS:
            raise ValueError(f"Il belief deve contenere {N_OPPONENTS} elementi")

    @property
    def map_id(self) -> int:
        """ID pubblico della mappa osservata."""
        return self.game_map.map_id

    @property
    def belief_array(self) -> np.ndarray:
        """Copia float64 del posterior, pronta per ulteriori elaborazioni."""
        return np.asarray(self.belief, dtype=np.float64).copy()


class POMDPGame:
    """Ambiente con stato informativo (X_t, b_t), senza rivelare K all'Agent.

    ``GridGame`` continua a simulare il contesto reale (m, k), ma e' una
    dipendenza privata. L'interfaccia pubblica di questo wrapper espone
    esclusivamente stato visibile, mappa, belief, azioni legali, eventi
    osservabili, reward ed esito del gioco.

    Un'istanza gestisce un solo episodio alla volta. Per ambienti paralleli
    creare istanze distinte, ciascuna con il proprio belief.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        shaping_lambda: float = 0.1,
        context_distribution: ContextDistribution = DEFAULT_CONTEXT_DISTRIBUTION,
        belief_prior: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        """Costruisce simulatore e filtro bayesiano.

        Il ``belief_prior`` viene configurato INDIPENDENTEMENTE da
        ``context_distribution``. Se None, e' uniforme sulle sei policy;
        questo evita di rivelare K durante la valutazione a contesto fisso.
        Per un vero prior non uniforme noto all'Agent, fornirlo esplicitamente.
        """
        self._game = GridGame(
            seed=seed,
            shaping_lambda=shaping_lambda,
            context_distribution=context_distribution,
        )
        self._belief_memory = BayesianOpponentBelief(prior=belief_prior)
        self._observed_opponent_action: Action | None = None

    @property
    def state(self) -> GameState | TerminalState:
        """Stato dinamico osservabile X_t o il simbolo terminale."""
        return self._game.state

    @property
    def game_map(self) -> GameMap:
        """Mappa dell'episodio, nota all'Agent; non espone l'avversario."""
        return self._game.context.game_map

    @property
    def map_id(self) -> int:
        return self.game_map.map_id

    @property
    def belief(self) -> np.ndarray:
        """Posterior corrente come copia float64 indipendente dalla memoria."""
        return self._belief_memory.probabilities

    @property
    def belief_prior(self) -> np.ndarray:
        """Prior configurato, mai ottenuto dal vero K di questo episodio."""
        return self._belief_memory.prior

    @property
    def terminated(self) -> bool:
        return self._game.terminated

    @property
    def outcome(self) -> GameOutcome | None:
        """Risultato pubblico, definito dopo l'attivazione numero 100."""
        return self._game.outcome

    @property
    def episode_seed(self) -> int:
        return self._game.episode_seed

    @property
    def shaping_lambda(self) -> float:
        return self._game.shaping_lambda

    def reset(
        self,
        *,
        seed: int | None = None,
        context: EpisodeContext | None = None,
        starting_player: Player | None = None,
    ) -> POMDPObservation:
        """Inizia una partita e reinizializza il belief al prior.

        Il contesto esplicito e' un parametro dell'*ambiente* destinato a
        test o evaluation: la policy vera non compare nell'osservazione.
        Non viene usato ``context.opponent_id`` per inizializzare il belief.
        """
        self._game.reset(
            seed=seed,
            context=context,
            starting_player=starting_player,
        )
        self._belief_memory.reset()
        self._observed_opponent_action = None
        return self.observe()

    def legal_actions(self) -> tuple[Action, ...]:
        """Azioni dell'Agent; DUMMY nei turni avversari o melee forzati."""
        return self._game.legal_actions()

    def observe(self) -> POMDPObservation:
        """Restituisce l'osservazione corrente, senza informazioni su K.

        Il terminale non ha una normale osservazione: dopo ``terminated``
        utilizzare ``outcome`` oppure chiamare ``reset``.
        """
        state = self._game.state
        if state is TERMINAL_STATE:
            raise RuntimeError("Il terminale non ha osservazione: chiamare reset()")
        if not isinstance(state, GameState):
            raise RuntimeError("Stato del simulatore non valido")
        return POMDPObservation(
            state=state,
            game_map=self.game_map,
            belief=tuple(float(p) for p in self._belief_memory.probabilities),
            observed_opponent_action=self._observed_opponent_action,
        )

    def step(self, action: Action | int) -> StepResult:
        """Esegue esattamente un'attivazione e aggiorna il belief online.

        Le transizioni di ``GridGame`` e i reward rimangono IDENTICI alla
        versione MDP. Il filtro legge soltanto lo stato PRE-azione e
        ``events.effective_action``. Il belief si aggiorna se e solo se
        l'Opponent ha realmente scelto un'azione (non nel melee forzato).

        Il respawn NON resetta il belief. Nemmeno la transizione finale lo
        resetta: il reset avviene esclusivamente con ``reset()``.

        Ritorna il normale ``StepResult`` del simulatore: contiene stati,
        reward ed eventi osservabili, ma nessun identificativo di K. Il
        posterior successivo e' ottenibile con ``belief`` o ``observe()``
        quando la transizione non e' terminale.
        """
        result = self._game.step(action)
        self._belief_memory.observe_step(result, self.game_map)

        previous = result.state
        if (
            previous.player_to_act is Player.OPPONENT
            and not are_adjacent(
                previous.agent_position, previous.opponent_position
            )
        ):
            observed = result.events.effective_action
            if observed is None or observed is Action.DUMMY:
                raise RuntimeError(
                    "Turno decisionale Opponent senza azione osservabile"
                )
            self._observed_opponent_action = observed
        else:
            self._observed_opponent_action = None

        return result


__all__: Final[tuple[str, ...]] = (
    "POMDPObservation",
    "POMDPGame",
)
