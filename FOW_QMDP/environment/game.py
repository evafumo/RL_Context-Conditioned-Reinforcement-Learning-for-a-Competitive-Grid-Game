"""Simulatore dell'MDP competitivo, con 100 attivazioni individuali.

Dipende soltanto da rules.py, maps.py, objectives.py, opponents.py e context.py.
Ogni step rappresenta una singola attivazione, anche nei turni Opponent e
nel melee obbligatorio (azione formale DUMMY). Nessuna rete neurale richiesta.

Reward: per t < 100, lambda * (D_{t+1} - D_t); per t = 100,
U(D_final) - lambda * D_100, dove D_100 e' il differenziale *prima*
dell'ultima attivazione. I termini di shaping telescopizzano a U(D_final).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from random import Random
from typing import Callable, Final

from .context import (
    DEFAULT_CONTEXT_DISTRIBUTION,
    ContextDistribution,
    EpisodeContext,
    sample_context,
)
from .objectives import ObjectiveSet, generate_objectives, resolve_objective_capture
from .opponents import OpponentState, choose_opponent_action, legal_opponent_actions
from .rules import (
    AGENT_START,
    OPPONENT_START,
    Action,
    MeleeOutcome,
    Player,
    Position,
    ShotOutcome,
    MOVEMENT_ACTIONS,
    are_adjacent,
    can_shoot,
    legal_movement_actions,
    move_destination,
    other_player,
    resolve_melee,
    resolve_shot,
    respawn_positions,
    score_for_kill,
    score_for_objective,
)

HORIZON: Final[int] = 100


class TerminalState(Enum):
    """Unico stato assorbente s† del modello matematico."""

    TERMINAL = "terminal"


TERMINAL_STATE: Final[TerminalState] = TerminalState.TERMINAL


@dataclass(frozen=True, slots=True)
class GameState:
    """Stato *prima* di un'attivazione; D si ricava dai VP individuali.

    L'MDP minimale usa (pA, pO, G, D, tau, t). Conservare separatamente
    i due VP permette di mostrare il punteggio senza alterare la dinamica.
    """

    agent_position: Position
    opponent_position: Position
    active_objectives: ObjectiveSet
    agent_vp: int
    opponent_vp: int
    player_to_act: Player
    activation: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "active_objectives", frozenset(self.active_objectives))
        if type(self.agent_vp) is not int or type(self.opponent_vp) is not int:
            raise ValueError("I Victory Points devono essere interi")
        if not isinstance(self.player_to_act, Player):
            raise ValueError("player_to_act deve essere un Player")
        if type(self.activation) is not int or not 1 <= self.activation <= HORIZON:
            raise ValueError("L'attivazione deve essere compresa tra 1 e 100")

    @property
    def score_differential(self) -> int:
        """D = VP_A - VP_O, la variabile utilizzata nel report."""
        return self.agent_vp - self.opponent_vp


@dataclass(frozen=True, slots=True)
class GameOutcome:
    """Punteggio finale osservato dopo la risoluzione dell'attivazione 100."""

    agent_vp: int
    opponent_vp: int
    winner: Player | None

    @property
    def score_differential(self) -> int:
        return self.agent_vp - self.opponent_vp

    @property
    def utility(self) -> int:
        """U(D_final)=1 in caso di vittoria Agent; 0 per sconfitta/pareggio."""
        return int(self.score_differential > 0)


@dataclass(frozen=True, slots=True)
class ActivationEvents:
    """Eventi diagnostici di un'unica attivazione, nell'ordine di risoluzione.

    effective_action=None indica il melee obbligatorio di inizio turno.
    La politica Opponent e' interrogata soltanto per le sue decisioni normali.
    """

    effective_action: Action | None = None
    captured_objective: Position | None = None
    objectives_regenerated: bool = False
    shot_outcome: ShotOutcome | None = None
    melee_outcome: MeleeOutcome | None = None
    melee_attacker: Player | None = None
    killer: Player | None = None


@dataclass(frozen=True, slots=True)
class StepResult:
    """Transizione MDP completa, utilizzabile da rollout.py e dai test."""

    state: GameState
    action: Action  # Azione MDP ricevuta (DUMMY se forzata)
    next_state: GameState | TerminalState
    reward: float
    terminated: bool
    events: ActivationEvents
    outcome: GameOutcome | None = None

    @property
    def forced(self) -> bool:
        return self.action is Action.DUMMY


@dataclass(frozen=True, slots=True)
class EpisodeTrace:
    """Esito e transizioni di una partita completa."""

    context: EpisodeContext
    transitions: tuple[StepResult, ...]
    outcome: GameOutcome

    @property
    def total_reward(self) -> float:
        return sum(transition.reward for transition in self.transitions)


# Per la simulazione stand-alone una policy Agent riceve lo stato, le azioni
# legali e un RNG riproducibile. Non e' richiesta da step(), che implementa MDP.
AgentPolicy = Callable[[GameState, tuple[Action, ...], Random], Action | int]


class GridGame:
    """Coordinatore delle transizioni dell'ambiente, separato da PPO.

    Uso principale:
        game = GridGame(seed=42, shaping_lambda=0.1)
        state = game.reset()  # campiona contesto, obiettivi e iniziativa
        while not game.terminated:
            legal = game.legal_actions()
            action = game.rng.choice(legal) if legal != (Action.DUMMY,) else Action.DUMMY
            result = game.step(action)

    game.step() accetta DUMMY nei turni Opponent e nei melee obbligatori.
    In tali stati l'Agent NON sceglie una normale azione di gioco.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        shaping_lambda: float = 0.1,
        context_distribution: ContextDistribution = DEFAULT_CONTEXT_DISTRIBUTION,
    ) -> None:
        if isinstance(shaping_lambda, bool) or not isinstance(shaping_lambda, (float, int)):
            raise ValueError("shaping_lambda deve essere un numero finito >= 0")
        if not isfinite(shaping_lambda) or shaping_lambda < 0:
            raise ValueError("shaping_lambda deve essere finito e >= 0")
        if not isinstance(context_distribution, ContextDistribution):
            raise TypeError("context_distribution deve essere ContextDistribution")

        self.shaping_lambda = float(shaping_lambda)
        self.context_distribution = context_distribution

        # Generatore principale
        self._master_rng = Random(seed)
        self._episode_seed: int | None = None

        # Generatori separati per le diverse componenti
        self.rng_context = Random()
        self.rng_initiative = Random()
        self.rng_objectives = Random()
        self.rng_agent = Random()
        self.rng_opponent = Random()
        self.rng_combat = Random()

        # Compatibilità con il vecchio codice
        self.rng = self.rng_agent
        
        self._context: EpisodeContext | None = None
        self._state: GameState | TerminalState | None = None
        self._outcome: GameOutcome | None = None


    @property
    def episode_seed(self) -> int:
        """Restituisce il seed dell'episodio corrente."""
        if self._episode_seed is None:
            raise RuntimeError(
                "Chiamare reset() prima di leggere episode_seed"
            )
        return self._episode_seed

    

    @property
    def context(self) -> EpisodeContext:
        if self._context is None:
            raise RuntimeError("Chiamare reset() prima di usare il contesto")
        return self._context

    @property
    def state(self) -> GameState | TerminalState:
        if self._state is None:
            raise RuntimeError("Chiamare reset() prima di accedere allo stato")
        return self._state

    @property
    def terminated(self) -> bool:
        return self._state is TERMINAL_STATE

    @property
    def outcome(self) -> GameOutcome | None:
        """None fino alla completa risoluzione della centesima attivazione."""
        return self._outcome

    def reset(
        self,
        *,
        seed: int | None = None,
        context: EpisodeContext | None = None,
        starting_player: Player | None = None,
    ) -> GameState:
        """Inizializza una nuova partita e restituisce S_1.

        - seed: se fornito, riavvia l'RNG; altrimenti continua lo stream.
        - context: se assente campiona (m,k); utile fissarlo per evaluation.
        - starting_player: override per test; se assente lancia un d6 equo.
        Le estrazioni avvengono nell'ordine contesto, obiettivi, iniziativa.
        """
        if context is not None and not isinstance(context, EpisodeContext):
            raise TypeError("context deve essere EpisodeContext o None")
        if starting_player is not None and not isinstance(starting_player, Player):
            raise TypeError("starting_player deve essere un Player o None")
        # Derivazione del seed dell'episodio
        episode_seed = (
            seed if seed is not None
            else self._master_rng.getrandbits(64)
        )

        # Generatore temporaneo per derivare sei seed
        stream_seeder = Random(episode_seed)

        for stream in (
            self.rng_context,
            self.rng_initiative,
            self.rng_objectives,
            self.rng_agent,
            self.rng_opponent,
            self.rng_combat,
        ):
            stream.seed(stream_seeder.getrandbits(64))

        self._episode_seed = episode_seed

        # Campionamento del contesto
        selected = context if context is not None else sample_context(
            self.rng_context,
            self.context_distribution,
        )

        # Generazione degli obiettivi
        objectives = generate_objectives(
            selected.game_map,
            AGENT_START,
            OPPONENT_START,
            self.rng_objectives,
        )

        # Iniziativa
        if starting_player is None:
            die = self.rng_initiative.randint(1, 6)
            starting_player = (
                Player.AGENT if die <= 3 else Player.OPPONENT
            )




        
        self._context = selected
        self._outcome = None
        self._state = GameState(
            agent_position=AGENT_START,
            opponent_position=OPPONENT_START,
            active_objectives=objectives,
            agent_vp=0,
            opponent_vp=0,
            player_to_act=starting_player,
            activation=1,
        )
        return self._state

    def legal_actions(self) -> tuple[Action, ...]:
        """Insieme A_m(S_t) visto dall'Agent, incluso il simbolo DUMMY.

        In un turno Opponent o in melee obbligatorio l'unica azione e' DUMMY.
        In una normale attivazione Agent include Pass, mosse legali e Shoot.
        """
        state = self.state
        if state is TERMINAL_STATE:
            return (Action.DUMMY,)  # stato assorbente; step() non prosegue
        if state.player_to_act is Player.OPPONENT or are_adjacent(
            state.agent_position, state.opponent_position
        ):
            return (Action.DUMMY,)

        moves = legal_movement_actions(
            state.agent_position,
            self.context.game_map.obstacles,
            state.opponent_position,
        )
        shooting = (Action.SHOOT,) if can_shoot(
            state.agent_position, state.opponent_position
        ) else ()
        return moves + shooting + (Action.PASS,)

    @staticmethod
    def _coerce_action(action: Action | int) -> Action:
        if isinstance(action, Action):
            return action
        if type(action) is int:
            try:
                return Action(action)
            except ValueError as exc:
                raise ValueError(f"ID azione non valido: {action}") from exc
        raise ValueError(f"Azione non valida: {action!r}")

    @staticmethod
    def _opponent_state(state: GameState) -> OpponentState:
        return OpponentState(
            agent_position=state.agent_position,
            opponent_position=state.opponent_position,
            active_objectives=state.active_objectives,
            score_differential=state.score_differential,
            player_to_act=state.player_to_act,
            activation=state.activation,
        )

    def step(self, action: Action | int) -> StepResult:
        """Esegue *esattamente una* attivazione e restituisce (s,a,r,s',done).

        Valida le azioni prima di consumare casualita'. In un turno Opponent
        richiede DUMMY, quindi campiona la policy Opponent corrente. Se il
        melee e' obbligatorio non interroga alcuna policy.
        """
        previous = self.state
        if previous is TERMINAL_STATE:
            raise RuntimeError("Partita terminata: chiamare reset()")

        requested = self._coerce_action(action)
        admissible = self.legal_actions()
        if requested not in admissible:
            raise ValueError(
                f"Azione {requested.name} illegale all'attivazione "
                f"{previous.activation}; azioni ammesse: "
                f"{tuple(item.name for item in admissible)}"
            )

        actor = previous.player_to_act
        agent_pos = previous.agent_position
        opponent_pos = previous.opponent_position
        goals = previous.active_objectives
        agent_vp = previous.agent_vp
        opponent_vp = previous.opponent_vp

        executed: Action | None = None
        captured: Position | None = None
        regenerated = False
        shot: ShotOutcome | None = None
        melee: MeleeOutcome | None = None
        melee_attacker: Player | None = None
        killer: Player | None = None

        # Il melee di inizio turno e' obbligatorio e non richiede alcuna azione.
        if are_adjacent(agent_pos, opponent_pos):
            melee_attacker = actor
            melee = resolve_melee(self.rng_combat)
        else:
            if actor is Player.OPPONENT:
                opponent_view = self._opponent_state(previous)
                executed = choose_opponent_action(
                    self.context.opponent,
                    opponent_view,
                    self.context.game_map,
                    self.rng_opponent,
                )
                # Una policy difettosa non deve mai violare le regole del gioco.
                if executed not in legal_opponent_actions(
                    opponent_view, self.context.game_map
                ):
                    raise RuntimeError(
                        f"La policy Opponent ha scelto un'azione illegale: {executed!r}"
                    )
            else:
                executed = requested

            if executed in MOVEMENT_ACTIONS:
                if actor is Player.AGENT:
                    agent_pos = move_destination(agent_pos, executed)
                else:
                    opponent_pos = move_destination(opponent_pos, executed)

                update = resolve_objective_capture(
                    game_map=self.context.game_map,
                    active_objectives=goals,
                    moving_player=actor,
                    agent_position=agent_pos,
                    opponent_position=opponent_pos,
                    rng=self.rng_objectives,
                )
                goals = update.active_objectives
                captured = update.captured_position
                regenerated = update.regenerated
                if update.captured:
                    delta = score_for_objective(actor)
                    agent_vp += delta.agent_vp
                    opponent_vp += delta.opponent_vp

                # Melee immediato anche dopo una cattura e una rigenerazione.
                if are_adjacent(agent_pos, opponent_pos):
                    melee_attacker = actor
                    melee = resolve_melee(self.rng_combat)
            elif executed is Action.SHOOT:
                shot = resolve_shot(self.rng_combat)
                if shot is ShotOutcome.TARGET_KILLED:
                    killer = actor
            elif executed is Action.PASS:
                pass
            else:
                # Un'azione illegale qui indica una violazione interna della
                # policy avversaria: non modificare silenziosamente le regole.
                raise RuntimeError(f"La policy Opponent ha scelto {executed!r}")

        if melee is MeleeOutcome.ATTACKER_KILLS:
            killer = melee_attacker
        elif melee is MeleeOutcome.DEFENDER_KILLS:
            assert melee_attacker is not None
            killer = other_player(melee_attacker)

        if killer is not None:
            delta = score_for_kill(killer)
            agent_vp += delta.agent_vp
            opponent_vp += delta.opponent_vp
            agent_pos, opponent_pos = respawn_positions()

        final_d = agent_vp - opponent_vp
        if previous.activation == HORIZON:
            winner = (
                Player.AGENT if final_d > 0 else
                Player.OPPONENT if final_d < 0 else None
            )
            outcome: GameOutcome | None = GameOutcome(
                agent_vp=agent_vp,
                opponent_vp=opponent_vp,
                winner=winner,
            )
            # Phi(s†)=0; non aggiungere lambda*(D_final-D_before).
            reward = float(outcome.utility - self.shaping_lambda * previous.score_differential)
            next_state: GameState | TerminalState = TERMINAL_STATE
        else:
            outcome = None
            reward = self.shaping_lambda * (
                final_d - previous.score_differential
            )
            next_state = GameState(
                agent_position=agent_pos,
                opponent_position=opponent_pos,
                active_objectives=goals,
                agent_vp=agent_vp,
                opponent_vp=opponent_vp,
                player_to_act=other_player(actor),
                activation=previous.activation + 1,
            )

        events = ActivationEvents(
            effective_action=executed,
            captured_objective=captured,
            objectives_regenerated=regenerated,
            shot_outcome=shot,
            melee_outcome=melee,
            melee_attacker=melee_attacker,
            killer=killer,
        )
        result = StepResult(
            state=previous,
            action=requested,
            next_state=next_state,
            reward=reward,
            terminated=outcome is not None,
            events=events,
            outcome=outcome,
        )
        self._state = next_state
        self._outcome = outcome
        return result

    def play_episode(
        self,
        agent_policy: AgentPolicy | None = None,
        *,
        seed: int | None = None,
        context: EpisodeContext | None = None,
        starting_player: Player | None = None,
    ) -> EpisodeTrace:
        """Simula una partita di 100 attivazioni senza reti neurali.

        Se agent_policy e' None, Agent sceglie uniformemente tra le proprie
        azioni legali. Per azioni forzate passa sempre DUMMY. Restituisce una
        traccia completa; i parametri PPO non vengono mai modificati.
        """
        self.reset(seed=seed, context=context, starting_player=starting_player)
        episode_context = self.context
        transitions: list[StepResult] = []
        for _ in range(HORIZON):
            state = self.state
            assert isinstance(state, GameState)
            legal = self.legal_actions()
            if legal == (Action.DUMMY,):
                action: Action | int = Action.DUMMY
            elif agent_policy is None:
                action = self.rng_agent.choice(legal)
            else:
                action = agent_policy(state, legal, self.rng_agent)
            transitions.append(self.step(action))
        assert self._outcome is not None
        return EpisodeTrace(
            context=episode_context,
            transitions=tuple(transitions),
            outcome=self._outcome,
        )
