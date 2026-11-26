"""Contatori diagnostici per le metriche C; nessuna modifica all'MDP.

Un accumulatore per ambiente/episodio, attivato soltanto nella valutazione.
Registra gli eventi gia' risolti da GridGame.step, non ricalcola regole,
combattimenti o catture. Nessun tensore PyTorch e nessuna comunicazione IPC
per singola attivazione: il riepilogo viene trasmesso solo a fine episodio.
"""
from __future__ import annotations

from dataclasses import dataclass

from environment.game import HORIZON, GameOutcome, StepResult
from environment.rules import (
    Action, MeleeOutcome, Player, ShotOutcome, OBJECTIVE_VP, KILLER_VP, VICTIM_VP,
)


@dataclass(slots=True)
class EpisodeEventCounters:
    """Statistiche di UNA partita. Creare un nuovo oggetto dopo ogni reset."""

    activations: int = 0
    objectives_agent: int = 0
    objectives_opponent: int = 0
    shots_agent: int = 0
    shots_opponent: int = 0
    shot_kills_agent: int = 0
    shot_kills_opponent: int = 0
    melee_encounters: int = 0
    melee_initiated_agent: int = 0
    melee_initiated_opponent: int = 0
    melee_kills_agent: int = 0
    melee_kills_opponent: int = 0
    kills_agent: int = 0
    kills_opponent: int = 0

    def record(self, transition: StepResult) -> None:
        """Legge un evento del simulatore DOPO la sua risoluzione."""
        if transition.state.activation != self.activations + 1:
            raise RuntimeError("Attivazioni non consecutive nel contatore diagnostico")
        self.activations += 1
        actor = transition.state.player_to_act
        ev = transition.events

        if ev.captured_objective is not None:
            if ev.effective_action not in (Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST):
                raise RuntimeError("Cattura senza movimento")
            if actor is Player.AGENT:
                self.objectives_agent += 1
            else:
                self.objectives_opponent += 1

        if ev.shot_outcome is not None:
            if ev.effective_action is not Action.SHOOT or ev.melee_outcome is not None:
                raise RuntimeError("Evento SHOOT incoerente")
            if actor is Player.AGENT:
                self.shots_agent += 1
                if ev.shot_outcome is ShotOutcome.TARGET_KILLED:
                    self.shot_kills_agent += 1
            else:
                self.shots_opponent += 1
                if ev.shot_outcome is ShotOutcome.TARGET_KILLED:
                    self.shot_kills_opponent += 1

        if ev.melee_outcome is not None:
            attacker = ev.melee_attacker
            if attacker not in (Player.AGENT, Player.OPPONENT):
                raise RuntimeError("Melee senza attaccante")
            self.melee_encounters += 1
            if attacker is Player.AGENT:
                self.melee_initiated_agent += 1
            else:
                self.melee_initiated_opponent += 1
            if ev.melee_outcome is MeleeOutcome.ATTACKER_KILLS:
                if attacker is Player.AGENT:
                    self.melee_kills_agent += 1
                else:
                    self.melee_kills_opponent += 1
            elif ev.melee_outcome is MeleeOutcome.DEFENDER_KILLS:
                if attacker is Player.AGENT:
                    self.melee_kills_opponent += 1
                else:
                    self.melee_kills_agent += 1
            elif ev.melee_outcome is not MeleeOutcome.BOTH_SURVIVE:
                raise RuntimeError("Esito melee sconosciuto")
        elif ev.melee_attacker is not None:
            raise RuntimeError("Attaccante melee senza evento")

        expected_killer = None
        if ev.shot_outcome is ShotOutcome.TARGET_KILLED:
            expected_killer = actor
        elif ev.melee_outcome is MeleeOutcome.ATTACKER_KILLS:
            expected_killer = ev.melee_attacker
        elif ev.melee_outcome is MeleeOutcome.DEFENDER_KILLS:
            expected_killer = (Player.OPPONENT if ev.melee_attacker is Player.AGENT
                               else Player.AGENT)
        if ev.killer is not expected_killer:
            raise RuntimeError("Killer non coerente con evento SHOOT/melee")
        if ev.killer is Player.AGENT:
            self.kills_agent += 1
        elif ev.killer is Player.OPPONENT:
            self.kills_opponent += 1

    def finish(self, outcome: GameOutcome) -> dict[str, object]:
        """Ritorna dati serializzabili; valida i contatori sul punteggio finale."""
        if self.activations != HORIZON:
            raise RuntimeError("Riepilogo richiesto prima di 100 attivazioni")
        if self.shot_kills_agent + self.melee_kills_agent != self.kills_agent:
            raise RuntimeError("Kill Agent non coerenti")
        if self.shot_kills_opponent + self.melee_kills_opponent != self.kills_opponent:
            raise RuntimeError("Kill Opponent non coerenti")
        if self.melee_initiated_agent + self.melee_initiated_opponent != self.melee_encounters:
            raise RuntimeError("Melee non coerenti")
        # Verifica indipendente sui VP usando le costanti del regolamento.
        expected_agent_vp = (OBJECTIVE_VP * self.objectives_agent
                             + KILLER_VP * self.kills_agent
                             + VICTIM_VP * self.kills_opponent)
        expected_opponent_vp = (OBJECTIVE_VP * self.objectives_opponent
                                + KILLER_VP * self.kills_opponent
                                + VICTIM_VP * self.kills_agent)
        if (outcome.agent_vp != expected_agent_vp or
                outcome.opponent_vp != expected_opponent_vp):
            raise RuntimeError("Statistiche eventi incoerenti con Victory Points")
        return {
            'activations': self.activations,
            'objective_captures': {
                'agent': self.objectives_agent,
                'opponent': self.objectives_opponent,
            },
            'combat': {
                'shots_agent': self.shots_agent,
                'shots_opponent': self.shots_opponent,
                'shot_kills_agent': self.shot_kills_agent,
                'shot_kills_opponent': self.shot_kills_opponent,
                'melee_encounters': self.melee_encounters,
                'melee_initiated_agent': self.melee_initiated_agent,
                'melee_initiated_opponent': self.melee_initiated_opponent,
                'melee_kills_agent': self.melee_kills_agent,
                'melee_kills_opponent': self.melee_kills_opponent,
                'kills_agent': self.kills_agent,
                'kills_opponent': self.kills_opponent,
                'deaths_agent': self.kills_opponent,
                'deaths_opponent': self.kills_agent,
            },
            'final_vp': {'agent': outcome.agent_vp, 'opponent': outcome.opponent_vp},
        }


__all__ = ['EpisodeEventCounters']
