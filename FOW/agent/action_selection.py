"""Selezione delle azioni per Context-Conditioned Masked DQN.

    QNetwork.forward(spatial, features) -> Q-value grezzi (B, 7).
    action_mask.mask_q_values(q_values, action_masks) -> Q mascherati.
    action_masks: torch.bool (B, 7), True = azione legale.
    DUMMY (ID 6) e' l'unica azione ammessa nei turni forzati.

Questo modulo implementa la policy comportamentale epsilon-greedy e la
valutazione greedy. NON implementa Double DQN, la target network, la loss,
la schedulazione di epsilon o le regole dell'MDP.

Le maschere devono provenire da environment.action_mask e devono essere
validate su CPU prima del trasferimento CUDA. Le funzioni qui sotto verificano
shape, dtype e device senza sincronizzare la GPU per ispezionare i dati.
In particolare, una maschera priva di azioni legali non e' un input valido.

Le funzioni restituiscono azioni int64 sul device dei Q-value: il collector
potra' trasferire un intero batch di ID a CPU per chiamare GridGame.step().
"""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor, nn

from environment.action_mask import NUM_ACTIONS, mask_q_values


def _validate_epsilon(epsilon: float) -> float:
    """Restituisce epsilon come float, imponendo 0 <= epsilon <= 1."""
    if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
        raise TypeError("epsilon deve essere un numero reale, non booleano")
    value = float(epsilon)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("epsilon deve essere finito e compreso fra 0 e 1")
    return value


def _validate_generator(generator: torch.Generator | None, device: torch.device) -> None:
    """Impedisce generatori incompatibili con il dispositivo dei Q-value."""
    if generator is None:
        return
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator deve essere torch.Generator oppure None")
    generator_device = generator.device
    if generator_device.type != device.type:
        raise ValueError("Il generatore RNG deve usare lo stesso tipo di device dei Q-value")
    if (device.type == "cuda" and generator_device.index is not None
            and device.index is not None and generator_device.index != device.index):
        raise ValueError("Il generatore RNG CUDA deve usare la stessa GPU dei Q-value")


def _masked_batch(q_values: Tensor, action_masks: Tensor) -> Tensor:
    """Valida il contratto (B,7) e usa il masking autorevole della Fase 2."""
    if not isinstance(q_values, Tensor):
        raise TypeError("q_values deve essere un tensore PyTorch")
    if q_values.ndim != 2 or q_values.shape[0] < 1 or q_values.shape[1] != NUM_ACTIONS:
        raise ValueError(f"q_values deve avere shape (B,{NUM_ACTIONS}) con B >= 1")
    # mask_q_values verifica anche dtype, device e shape della maschera.
    return mask_q_values(q_values, action_masks)


@torch.no_grad()
def greedy_actions(q_values: Tensor, action_masks: Tensor) -> Tensor:
    """Seleziona argmax sui soli Q-value legali, output torch.long (B,).

    A parita' di Q-value seleziona l'azione legale con ID piu' piccolo,
    seguendo il comportamento deterministico di torch.argmax.
    Si usa anche per la valutazione (epsilon=0).
    """
    masked = _masked_batch(q_values, action_masks)
    return masked.argmax(dim=1)


@torch.no_grad()
def epsilon_greedy_actions(
    q_values: Tensor,
    action_masks: Tensor,
    epsilon: float,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Campiona una policy epsilon-greedy fra le sole azioni legali.

    Args:
        q_values: Q-value NON mascherati (B, 7), float, CPU oppure CUDA.
        action_masks: tensor torch.bool (B, 7), stesso device dei Q-value,
            ottenuto dalle funzioni di environment.action_mask.
        epsilon: probabilita' di selezione casuale, reale in [0, 1].
        generator: opzionale torch.Generator sullo stesso tipo di device,
            utile per rendere ripetibile il campionamento.

    Returns:
        Tensor torch.long (B,) sullo stesso device dei Q-value. L'azione
        selezionata e' sempre legale se le maschere rispettano il contratto.

    Policy:
        epsilon=0: greedy senza consumare RNG.
        epsilon=1: uniforme tra le azioni legali.
        0<epsilon<1: esplora con probabilita' epsilon, altrimenti greedy.

    DUMMY: in un turno forzato la sua maschera contiene un solo True;
    pertanto sia il ramo greedy sia quello casuale restituiscono DUMMY.
    Nessuna scelta esplorativa effettiva e' effettuata su questi turni.
    Per preservare la vettorizzazione CUDA, il campionamento puo' comunque
    consumare numeri casuali anche per le righe DUMMY quando epsilon > 0.
    Il conteggio delle decisioni esplorabili e' responsabilita' del collector.

    Non modifica q_values, action_masks, i parametri della rete o l'MDP.
    """
    probability = _validate_epsilon(epsilon)
    masked = _masked_batch(q_values, action_masks)
    _validate_generator(generator, q_values.device)
    if probability == 0.0:
        return masked.argmax(dim=1)

    # Multinomial su pesi binari: ogni azione legale ha uguale probabilita'.
    # Operazioni tutte sul device originario, senza ciclo Python sul batch.
    random_actions = torch.multinomial(
        action_masks.to(dtype=torch.float32),
        num_samples=1,
        replacement=True,
        generator=generator,
    ).squeeze(1)
    if probability == 1.0:
        return random_actions

    greedy = masked.argmax(dim=1)
    explore = torch.rand(
        (q_values.shape[0],),
        device=q_values.device,
        generator=generator,
    ) < probability
    return torch.where(explore, random_actions, greedy)


@torch.no_grad()
def select_actions(
    network: nn.Module,
    spatial: Tensor,
    features: Tensor,
    action_masks: Tensor,
    epsilon: float,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Inferenza batched della Q-network seguita da epsilon-greedy mascherato.

    Args:
        network: QNetwork o un nn.Module con forward(spatial, features)
            che restituisce Q-value grezzi (B, 7).
        spatial: tensor floating (B,4,9,9), sul device della rete.
        features: tensor floating (B,13), sul medesimo device.
        action_masks: tensor bool (B,7), sul medesimo device.
        epsilon: probabilita' di esplorazione in [0,1].
        generator: RNG opzionale sul device dei Q-value.

    Returns:
        Azioni torch.long (B,), sul device della rete.

    La funzione non cambia network.train()/network.eval(), non aggiorna
    parametri e non costruisce un grafo autograd. Il chiamante gestisce il
    trasferimento CPU->GPU e la conversione degli ID in Action per l'MDP.
    """
    if not isinstance(network, nn.Module):
        raise TypeError("network deve essere un torch.nn.Module")
    _validate_epsilon(epsilon)
    q_values = network(spatial, features)
    return epsilon_greedy_actions(
        q_values, action_masks, epsilon, generator=generator
    )


__all__ = ["greedy_actions", "epsilon_greedy_actions", "select_actions"]
