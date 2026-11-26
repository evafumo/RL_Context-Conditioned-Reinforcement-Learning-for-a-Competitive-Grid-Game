"""Action masking per Context-Conditioned Masked DQN.

Fonte unica delle azioni legali: GridGame.legal_actions(). Questo modulo
NON ricostruisce le regole di movimento, tiro o melee e NON altera l'MDP.

Convenzioni:
    Action.NORTH .. Action.DUMMY -> indici 0 .. 6 (7 azioni).
    maschera NumPy: bool, shape (7,) o (B, 7).
    True = azione legale, False = azione illegale.
    In un turno forzato la sola azione legale e' DUMMY (indice 6).
    Uno stato terminale non ha un'azione selezionabile: non creare una
    maschera terminale. Nel DQN il flag 'done' azzera il bootstrap.

PyTorch viene importato solo dalle funzioni che lo richiedono. I worker CPU
possono quindi generare maschere senza inizializzare CUDA. Trasferire batch
contigui alla GPU, preferibilmente insieme ai batch di osservazioni.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Iterable, Sequence

import numpy as np

from .game import GridGame
from .rules import Action

if TYPE_CHECKING:
    import torch


NUM_ACTIONS: Final[int] = len(Action)
DUMMY_ACTION_ID: Final[int] = int(Action.DUMMY)


def legal_actions_to_mask(legal_actions: Iterable[Action | int]) -> np.ndarray:
    """Codifica una lista *gia' calcolata* di azioni legali.

    Args:
        legal_actions: azioni provenienti da GridGame.legal_actions();
            ammette ID interi Python 0..6 oppure valori Action.

    Returns:
        ndarray NumPy booleano indipendente, shape (7,), C-contiguous.

    Raises:
        TypeError: argomento non iterabile.
        ValueError: sequenza vuota, ripetizioni, ID non validi o combinazioni
            incoerenti (DUMMY insieme ad altre azioni, o mancanza di PASS
            in un turno ordinario).

    Non interpreta la geometria: e' soltanto un adattatore di formato.
    """
    try:
        actions = tuple(legal_actions)
    except TypeError as exc:
        raise TypeError("legal_actions deve essere iterabile") from exc

    if not actions:
        raise ValueError("Almeno un'azione deve essere legale")

    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    for action in actions:
        if not isinstance(action, Action) and type(action) is not int:
            raise ValueError(f"Tipo di azione non valido: {action!r}")
        try:
            action_id = int(Action(action))
        except ValueError as exc:
            raise ValueError(f"ID di azione sconosciuto: {action!r}") from exc
        if mask[action_id]:
            raise ValueError(f"Azione ripetuta: {Action(action_id).name}")
        mask[action_id] = True

    _validate_action_mask(mask)
    return mask


def legal_action_mask(game: GridGame) -> np.ndarray:
    """Restituisce il masking dello stato corrente NON terminale.

    Chiama esclusivamente game.legal_actions(): la legalita' non viene
    calcolata due volte. Non modifica stato del gioco, RNG o contesto.
    """
    if not isinstance(game, GridGame):
        raise TypeError("game deve essere un GridGame")
    if game.terminated:
        raise ValueError("Lo stato terminale non ha azioni selezionabili")
    return legal_actions_to_mask(game.legal_actions())


def _validate_action_mask(mask: np.ndarray) -> None:
    """Controlla il contratto per UNA maschera NumPy."""
    if not isinstance(mask, np.ndarray):
        raise TypeError("La maschera deve essere un ndarray NumPy")
    if mask.shape != (NUM_ACTIONS,) or mask.dtype != np.bool_:
        raise ValueError(f"Attesa maschera bool di shape ({NUM_ACTIONS},)")
    if not mask.any():
        raise ValueError("Maschera senza azioni legali")
    if mask[DUMMY_ACTION_ID]:
        if mask.sum() != 1:
            raise ValueError("DUMMY deve essere l'unica azione nei turni forzati")
    elif not mask[int(Action.PASS)]:
        raise ValueError("PASS e' sempre legale in un turno Agent non forzato")


def stack_action_masks(masks: Sequence[np.ndarray]) -> np.ndarray:
    """Raggruppa B maschere CPU in un ndarray bool di shape (B, 7).

    Conserva l'ordine, verifica ogni elemento e non condivide memoria con
    le maschere di input. Utile per inferenza di piu' ambienti CPU.
    """
    if not isinstance(masks, Sequence) or isinstance(masks, (str, bytes)):
        raise TypeError("masks deve essere una sequenza di ndarray")
    if len(masks) == 0:
        raise ValueError("Occorre almeno una maschera")
    for index, mask in enumerate(masks):
        try:
            _validate_action_mask(mask)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"Maschera {index}: {exc}") from exc
    return np.ascontiguousarray(np.stack(masks, axis=0))


def to_torch_action_masks(
    masks: np.ndarray,
    device: str | torch.device = "cpu",
    *,
    pin_memory: bool = False,
    non_blocking: bool = False,
) -> torch.Tensor:
    """Converte maschere NumPy bool (7,) o (B,7) in tensori PyTorch bool.

    Le maschere sono verificate su CPU PRIMA del trasferimento: si evitano
    sincronizzazioni CUDA a ogni inferenza. Con piu' ambienti, trasferire
    un batch usando device='cuda', pin_memory=True, non_blocking=True.
    """
    if not isinstance(masks, np.ndarray):
        raise TypeError("masks deve essere un ndarray NumPy")
    if masks.ndim == 1:
        _validate_action_mask(masks)
    elif masks.ndim == 2 and masks.shape[1] == NUM_ACTIONS:
        if masks.shape[0] == 0:
            raise ValueError("Un batch vuoto non e' ammesso")
        for row in masks:
            _validate_action_mask(row)
    else:
        raise ValueError(f"Attese maschere con shape ({NUM_ACTIONS},) o (B,{NUM_ACTIONS})")

    try:
        import torch
    except ImportError as exc:
        raise ImportError("Installare PyTorch per convertire le maschere") from exc

    cpu_masks = torch.from_numpy(np.ascontiguousarray(masks))
    if pin_memory:
        if not torch.cuda.is_available():
            raise RuntimeError("pin_memory=True richiede PyTorch con CUDA disponibile")
        cpu_masks = cpu_masks.pin_memory()
    return cpu_masks.to(device=device, non_blocking=non_blocking)


def mask_q_values(q_values: torch.Tensor, action_masks: torch.Tensor) -> torch.Tensor:
    """Sostituisce con -inf i Q-value di azioni illegali, SENZA mutare q_values.

    Args:
        q_values: tensor floating [7] o [B,7] della Q-network.
        action_masks: tensor bool di shape IDENTICA, sul medesimo device.
            Deve essere stato generato/validato con questo modulo.

    Returns:
        Tensor con stessa shape, dtype e device di q_values. Le azioni
        illegali valgono -inf: argmax/max le ignorano. Le componenti legali
        conservano i Q-value e i relativi gradienti.

    NOTA: applicare la maschera per scegliere un'azione o il massimo del
    target Bellman. Per calcolare Q_online(s,a), usare i Q-value originali.
    Nei terminali il termine bootstrap va azzerato tramite 'done'.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Installare PyTorch per usare mask_q_values()") from exc

    if not isinstance(q_values, torch.Tensor):
        raise TypeError("q_values deve essere un tensore PyTorch")
    if not isinstance(action_masks, torch.Tensor):
        raise TypeError("action_masks deve essere un tensore PyTorch")
    if q_values.ndim not in (1, 2) or q_values.shape[-1] != NUM_ACTIONS:
        raise ValueError("q_values deve avere shape (7,) o (B,7)")
    if q_values.shape != action_masks.shape:
        raise ValueError("Q-value e maschere devono avere shape identica")
    if not q_values.is_floating_point():
        raise TypeError("q_values deve essere floating point")
    if action_masks.dtype != torch.bool:
        raise TypeError("action_masks deve avere dtype torch.bool")
    if q_values.device != action_masks.device:
        raise ValueError("Q-value e maschere devono essere sullo stesso device")

    # Operazione vettorializzata GPU-friendly; nessun round-trip su CPU.
    return q_values.masked_fill(~action_masks, -torch.inf)
