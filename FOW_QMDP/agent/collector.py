"""Raccolta batched di transizioni per Context-Conditioned Masked DQN.

ATTENZIONE AL RESET AUTOMATICO: dopo done=True, step.next_observation e'
gia' del NUOVO episodio. Per il replay, il terminale e' invece s_dagger,
privo di osservazione: si passa esplicitamente next_observation=None e
next_action_mask=None. Sugli altri step si utilizza step.transition_next.

Nessun peso viene aggiornato qui; l'ottimizzazione e' in agent.dqn.
La schedulazione di epsilon, gli update e i checkpoint spettano a train.py.
Usare il processo principale per la Q-network CUDA; non inviare mai il
modello PyTorch ai worker. Windows richiede if __name__ == '__main__' per
avviare ParallelGridGames.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np

from agent.action_selection import select_actions
from agent.parallel_env import ParallelGridGames
from agent.replay_buffer import ReplayBuffer
from environment.observations import Observation, ObservationBatch, to_torch_batch
from environment.action_mask import to_torch_action_masks

if TYPE_CHECKING:
    import torch
    from torch import nn


@dataclass(frozen=True, slots=True)
class CollectionStats:
    """Statistiche della SOLA raccolta corrente (non di episodi incompleti).

    ``rewards_sum`` somma i reward di tutte le transizioni raccolte: se la
    chiamata inizia/finisce a meta' episodio, NON e' un ritorno episodico.
    ``agent_decisions`` esclude le attivazioni DUMMY; ``forced_activations``
    include anche le attivazioni dell'avversario e tutti i turni forzati.
    ``wins`` conta outcome.utility == 1 sui soli step terminali.
    """

    num_envs: int
    steps_per_env: int
    transitions: int
    agent_decisions: int
    forced_activations: int
    completed_episodes: int
    wins: int
    rewards_sum: float


def collect_parallel_experience(
    environments: ParallelGridGames,
    online_network: nn.Module,
    replay_buffer: ReplayBuffer,
    *,
    steps_per_env: int,
    epsilon: float,
    device: str | torch.device | None = None,
    pin_memory: bool = False,
    generator: torch.Generator | None = None,
) -> CollectionStats:
    """Raccoglie ``steps_per_env * num_envs`` transizioni in un replay CPU.

    Args:
        environments: ParallelGridGames gia' avviato tramite start() o with.
        online_network: nn.Module con forward(spatial, features)->(N,7),
            gia' collocato sul device desiderato. Per un DQNTrainer, passare
            ``trainer.online_network``; mai la target network.
        replay_buffer: memoria circolare da riempire nel processo principale.
        steps_per_env: attivazioni per ambiente, > 0, anche oltre un episodio.
        epsilon: probabilita' di azione casuale legale in [0,1]. Rimane
            COSTANTE in questa chiamata; lo schedule spetta a train.py.
        device: device dell'online_network, se specificato; per CUDA usare
            'cuda:0' oppure 'cuda' quando il modello usa la GPU predefinita.
        pin_memory: abilita la memoria CPU pinned per i trasferimenti H2D;
            richiede device CUDA e PyTorch CUDA. Il beneficio va misurato.
        generator: torch.Generator opzionale sullo stesso device della rete
            per riprodurre l'esplorazione epsilon-greedy. Il suo stato e'
            gestito esternamente per i checkpoint.

    Returns:
        CollectionStats relativa esattamente alle transizioni raccolte.

    Note:
        Ogni N osservazioni e le relative maschere sono trasferite in batch
        CPU->GPU, e le N azioni tornano in un'unica copia GPU->CPU. Nessuna
        backward(), optimizer.step() o update target avviene nel collector.

        Le azioni illegali sono prevenute da action_selection e validate di
        nuovo da environments.step(); il replay valida anche ogni inserzione.
        Per i terminali NON si passa mai la nuova osservazione post-reset.

        Se il ReplayBuffer ha capacity inferiore al numero di transizioni,
        sovrascrive in modo circolare le piu' vecchie (comportamento atteso).
    """
    import torch
    from torch import nn

    if not isinstance(environments, ParallelGridGames):
        raise TypeError("environments deve essere ParallelGridGames")
    if not isinstance(online_network, nn.Module):
        raise TypeError("online_network deve essere torch.nn.Module")
    if not isinstance(replay_buffer, ReplayBuffer):
        raise TypeError("replay_buffer deve essere ReplayBuffer")
    if type(steps_per_env) is not int or steps_per_env <= 0:
        raise ValueError("steps_per_env deve essere un int positivo (non bool)")
    if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
        raise TypeError("epsilon deve essere un numero reale, non bool")
    probability = float(epsilon)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("epsilon deve essere finito in [0,1]")
    if type(pin_memory) is not bool:
        raise TypeError("pin_memory deve essere bool")

    # Si verifica il device PRIMA di qualsiasi step per non alterare gli
    # ambienti in caso di errore di configurazione (per quanto possibile).
    params = tuple(online_network.parameters())
    if not params:
        raise ValueError("online_network deve avere almeno un parametro")
    model_device = params[0].device
    if any(p.device != model_device for p in params):
        raise ValueError("Tutti i parametri della rete devono stare sullo stesso device")
    if any(b.device != model_device for b in online_network.buffers()):
        raise ValueError("Tutti i buffer della rete devono stare sullo stesso device")
    selected_device = torch.device(device) if device is not None else model_device
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile")
    if selected_device.type == "cuda" and selected_device.index is None:
        # CUDA senza indice significa la GPU predefinita; il modello deve
        # essere sulla stessa GPU, non su una seconda GPU implicitamente.
        selected_device = torch.device("cuda", torch.cuda.current_device())
    if selected_device != model_device:
        raise ValueError("Il device richiesto non coincide con quello del modello")
    if pin_memory and selected_device.type != "cuda":
        raise ValueError("pin_memory=True e' previsto solo con device CUDA")
    if generator is not None:
        if not isinstance(generator, torch.Generator):
            raise TypeError("generator deve essere torch.Generator o None")
        if generator.device.type != selected_device.type:
            raise ValueError("Il generatore RNG non usa il device della rete")
        if (selected_device.type == "cuda" and generator.device.index is not None
                and generator.device.index != selected_device.index):
            raise ValueError("Il generatore RNG deve usare la stessa GPU")

    # Deve essere gia' avviato: la verifica avviene prima del cambio eval().
    _ = environments.observation

    decisions = forced = completed = wins = 0
    rewards_sum = 0.0
    # Preservare anche i flag training dei sotto-moduli, che potrebbero
    # essere diversi da quello della rete principale.
    training_flags = [(module, module.training) for module in online_network.modules()]
    try:
        online_network.eval()
        with torch.inference_mode():
            for _ in range(steps_per_env):
                obs = environments.observation  # snapshot PRE-step
                # Unico batch CPU->device per tutte le osservazioni.
                network_obs = to_torch_batch(
                    ObservationBatch(obs.spatial, obs.features),
                    selected_device,
                    pin_memory=pin_memory,
                    non_blocking=pin_memory,
                )
                network_masks = to_torch_action_masks(
                    obs.action_masks,
                    selected_device,
                    pin_memory=pin_memory,
                    non_blocking=pin_memory,
                )
                actions_tensor = select_actions(
                    online_network,
                    network_obs.spatial,
                    network_obs.features,
                    network_masks,
                    probability,
                    generator=generator,
                )
                # Un'unica copia CUDA->CPU e un'unica sincronizzazione batch.
                actions = np.ascontiguousarray(actions_tensor.cpu().numpy(), dtype=np.int64)
                result = environments.step(actions)

                # Un inserimento per OGNI attivazione, incluse quelle DUMMY.
                # L'Observation terminale e' None: impedisce qualsiasi
                # collegamento artificiale con il nuovo episodio post-reset.
                for i in range(environments.num_envs):
                    done = bool(result.dones[i])
                    next_obs = None if done else Observation(
                        result.transition_next.spatial[i],
                        result.transition_next.features[i],
                    )
                    next_mask = None if done else result.transition_next.action_masks[i]
                    replay_buffer.add(
                        observation=Observation(obs.spatial[i], obs.features[i]),
                        action_mask=obs.action_masks[i],
                        action=int(actions[i]),
                        reward=float(result.rewards[i]),
                        next_observation=next_obs,
                        next_action_mask=next_mask,
                        done=done,
                    )

                new_decisions = int(np.count_nonzero(obs.is_agent_decision))
                decisions += new_decisions
                forced += environments.num_envs - new_decisions
                completed += int(np.count_nonzero(result.dones))
                wins += int(np.count_nonzero(result.terminal_utilities == 1))
                rewards_sum += float(np.sum(result.rewards, dtype=np.float64))
    finally:
        # Ripristino esatto anche in reti con sotto-moduli in modalita' mista.
        for module, training in training_flags:
            module.training = training

    return CollectionStats(
        num_envs=environments.num_envs,
        steps_per_env=steps_per_env,
        transitions=environments.num_envs * steps_per_env,
        agent_decisions=decisions,
        forced_activations=forced,
        completed_episodes=completed,
        wins=wins,
        rewards_sum=rewards_sum,
    )


__all__ = ["CollectionStats", "collect_parallel_experience"]
