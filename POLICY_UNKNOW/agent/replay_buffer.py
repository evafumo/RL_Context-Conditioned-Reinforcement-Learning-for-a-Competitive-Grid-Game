"""Replay buffer uniforme per Context-Conditioned Masked DQN (standard).

    - Observation.spatial: (4, 9, 9), np.float32;
      Observation.features: (13,), np.float32.
    - Maschere correnti e successive: (7,), np.bool_, True = azione legale.
      La validazione usa environment.action_mask: NON ricalcola le regole.
    - Le sette azioni hanno ID 0..6, con DUMMY = 6.
    - Uno stato terminale non ha osservazione o maschera successive.

Il buffer memorizza TUTTE le transizioni, comprese quelle con DUMMY, in RAM
CPU preallocata. Un RNG NumPy privato estrae minibatch uniformi SENZA
rimpiazzo. Nessuna dipendenza CUDA viene inizializzata all'importazione:
solo to_torch_replay_batch() importa PyTorch, quando e' necessario.

In una transizione terminale, next_observation=None e next_action_mask=None
sono OBBLIGATORI. Internamente il buffer memorizza segnaposto tutti zero;
questi NON rappresentano uno stato dell'MDP e sono ignorati dal DQN perche'
dones=True. Il futuro agent/dqn.py deve annullare il bootstrap dei terminali.

Il buffer non esegue step(), non valuta Q(s,a), non calcola target Bellman
ne' decide quale azione selezionare. E' gestito nel processo principale:
i worker CPU dell'ambiente restituiscono transizioni, senza condividere
questa struttura mutabile tra processi.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from environment.action_mask import NUM_ACTIONS, _validate_action_mask
from environment.observations import FEATURE_SIZE, SPATIAL_SHAPE, Observation
from environment.rules import Action

if TYPE_CHECKING:
    import torch

REPLAY_STATE_VERSION: Final[int] = 1


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    """Minibatch indipendente dagli array interni del buffer, su CPU.

    Shapes, dove B e' il numero di transizioni campionate:
        spatial, next_spatial: (B, 4, 9, 9), float32
        features, next_features: (B, 13), float32
        action_masks, next_action_masks: (B, 7), bool
        actions: (B,), int64; rewards: (B,), float32; dones: (B,), bool

    Per le righe terminali, next_spatial/next_features/next_action_masks
    sono zero. Una maschera tutta False produce max=-inf: evitare di
    moltiplicarlo per zero (NaN); filtrare le righe terminali oppure
    usare torch.where(dones, 0, max_next_q) prima del target Bellman.
    """

    spatial: np.ndarray
    features: np.ndarray
    action_masks: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_spatial: np.ndarray
    next_features: np.ndarray
    next_action_masks: np.ndarray
    dones: np.ndarray

    @property
    def batch_size(self) -> int:
        """Numero di transizioni nel minibatch."""
        return int(self.actions.shape[0])


@dataclass(frozen=True, slots=True)
class TorchReplayBatch:
    """Minibatch convertito in tensori sul device selezionato."""

    spatial: torch.Tensor
    features: torch.Tensor
    action_masks: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_spatial: torch.Tensor
    next_features: torch.Tensor
    next_action_masks: torch.Tensor
    dones: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.actions.shape[0])


def _validate_observation(observation: Observation, name: str) -> None:
    """Verifica l'interfaccia numerica senza ricostruire lo stato del gioco."""
    if not isinstance(observation, Observation):
        raise TypeError(f"{name} deve essere una Observation")
    spatial, features = observation.spatial, observation.features
    if not isinstance(spatial, np.ndarray) or not isinstance(features, np.ndarray):
        raise TypeError(f"{name} deve contenere array NumPy")
    if (spatial.shape != SPATIAL_SHAPE or spatial.dtype != np.float32 or
            features.shape != (FEATURE_SIZE,) or features.dtype != np.float32):
        raise ValueError(
            f"{name}: attesi spatial {SPATIAL_SHAPE} e features "
            f"({FEATURE_SIZE},), entrambi float32"
        )
    if not np.isfinite(spatial).all() or not np.isfinite(features).all():
        raise ValueError(f"{name}: l'osservazione deve contenere valori finiti")


def _validate_action(action: Action | int) -> int:
    """Accetta Action e interi (anche NumPy), ma non booleani o float."""
    if isinstance(action, (bool, np.bool_)) or not isinstance(action, Integral):
        raise TypeError("action deve essere un ID intero o Action, non bool/float")
    action_id = int(action)
    if not 0 <= action_id < NUM_ACTIONS:
        raise ValueError(f"action deve essere in [0, {NUM_ACTIONS - 1}]")
    return action_id


def _validate_done(done: bool) -> bool:
    if not isinstance(done, (bool, np.bool_)):
        raise TypeError("done deve essere booleano")
    return bool(done)


class ReplayBuffer:
    """Memoria circolare uniforme, dedicata a un singolo collector centrale.

    Args:
        capacity: numero massimo di transizioni, intero > 0.
        seed: seed opzionale per RNG locale np.random.default_rng().

    Usare preferibilmente una capacita' configurabile in train.py.
    Esempio: capacity=100_000 impiega circa 260 MiB per gli array del buffer.
    Non spostare l'intero buffer sulla RTX: trasferire solo i minibatch.
    """

    def __init__(self, capacity: int, *, seed: int | None = None) -> None:
        if type(capacity) is not int:
            raise TypeError("capacity deve essere un intero, non bool")
        if capacity <= 0:
            raise ValueError("capacity deve essere maggiore di zero")
        if seed is not None and type(seed) is not int:
            raise TypeError("seed deve essere un intero oppure None")

        self.capacity = capacity
        self._rng = np.random.default_rng(seed)
        self._size = 0
        self._position = 0
        self._total_added = 0

        # Preallocazione CPU: il replay non occupa memoria CUDA.
        self._spatial = np.empty((capacity, *SPATIAL_SHAPE), dtype=np.float32)
        self._features = np.empty((capacity, FEATURE_SIZE), dtype=np.float32)
        self._action_masks = np.empty((capacity, NUM_ACTIONS), dtype=np.bool_)
        self._actions = np.empty(capacity, dtype=np.int64)
        self._rewards = np.empty(capacity, dtype=np.float32)
        self._next_spatial = np.empty((capacity, *SPATIAL_SHAPE), dtype=np.float32)
        self._next_features = np.empty((capacity, FEATURE_SIZE), dtype=np.float32)
        self._next_action_masks = np.empty((capacity, NUM_ACTIONS), dtype=np.bool_)
        self._dones = np.empty(capacity, dtype=np.bool_)

    def __len__(self) -> int:
        """Numero di transizioni effettivamente memorizzate."""
        return self._size

    @property
    def total_added(self) -> int:
        """Transizioni aggiunte dal costruttore o dall'ultimo clear()."""
        return self._total_added

    @property
    def position(self) -> int:
        """Prossima posizione di scrittura nell'anello [0, capacity)."""
        return self._position

    @property
    def nbytes(self) -> int:
        """Byte dei nove array preallocati, escluso l'overhead Python."""
        return sum(array.nbytes for array in self._arrays().values())

    def _arrays(self) -> dict[str, np.ndarray]:
        """Vista sulle colonne di storage per snapshot e ripristino."""
        return {
            "spatial": self._spatial,
            "features": self._features,
            "action_masks": self._action_masks,
            "actions": self._actions,
            "rewards": self._rewards,
            "next_spatial": self._next_spatial,
            "next_features": self._next_features,
            "next_action_masks": self._next_action_masks,
            "dones": self._dones,
        }

    def add(
        self,
        *,
        observation: Observation,
        action_mask: np.ndarray,
        action: Action | int,
        reward: float,
        next_observation: Observation | None,
        next_action_mask: np.ndarray | None,
        done: bool,
    ) -> None:
        """Inserisce UNA transizione verificata e aggiorna il cursore.

        La maschera corrente deve provenire da legal_action_mask(game).
        L'azione deve essere consentita dalla maschera, anche se e' DUMMY.
        Per done=False sono necessari osservazione e maschera successive.
        Per done=True entrambe devono essere None (terminale assorbente).
        Tutta la validazione precede la scrittura: input errati non cambiano
        size, cursore, contenuto del buffer o RNG.

        Gli array d'origine sono copiati: il riutilizzo dei buffer del
        collector non puo' alterare le transizioni storiche.
        """
        _validate_observation(observation, "observation")
        _validate_action_mask(action_mask)
        action_id = _validate_action(action)
        if not action_mask[action_id]:
            raise ValueError(f"L'azione {action_id} non e' legale nella action_mask")
        if isinstance(reward, (bool, np.bool_)) or not isinstance(reward, Real):
            raise TypeError("reward deve essere un numero reale")
        reward64 = float(reward)
        if not math.isfinite(reward64):
            raise ValueError("reward deve essere finito")
        if abs(reward64) > float(np.finfo(np.float32).max):
            raise ValueError("reward non e' rappresentabile come float32 finito")
        reward32 = np.float32(reward64)
        terminal = _validate_done(done)

        if terminal:
            if next_observation is not None or next_action_mask is not None:
                raise ValueError(
                    "done=True: next_observation e next_action_mask devono essere None"
                )
        else:
            if next_observation is None or next_action_mask is None:
                raise ValueError(
                    "done=False: next_observation e next_action_mask sono obbligatori"
                )
            _validate_observation(next_observation, "next_observation")
            _validate_action_mask(next_action_mask)

        index = self._position
        self._spatial[index] = observation.spatial
        self._features[index] = observation.features
        self._action_masks[index] = action_mask
        self._actions[index] = action_id
        self._rewards[index] = reward32
        if terminal:
            self._next_spatial[index].fill(0.0)
            self._next_features[index].fill(0.0)
            self._next_action_masks[index].fill(False)
        else:
            assert next_observation is not None and next_action_mask is not None
            self._next_spatial[index] = next_observation.spatial
            self._next_features[index] = next_observation.features
            self._next_action_masks[index] = next_action_mask
        self._dones[index] = terminal
        self._position = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._total_added += 1

    def sample(self, batch_size: int) -> ReplayBatch:
        """Estrae B indici uniformi SENZA rimpiazzo dalle celle occupate.

        Args:
            batch_size: numero positivo di transizioni, <= len(buffer).

        Returns:
            ReplayBatch con array indipendenti e C-contiguous;
            l'ordine e' quello casuale di estrazione, non temporale.
        """
        if type(batch_size) is not int:
            raise TypeError("batch_size deve essere un intero, non bool")
        if batch_size <= 0:
            raise ValueError("batch_size deve essere maggiore di zero")
        if batch_size > self._size:
            raise ValueError(
                f"Batch richiesto {batch_size}, transizioni disponibili {self._size}"
            )
        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        return ReplayBatch(**{
            name: np.ascontiguousarray(array[indices])
            for name, array in self._arrays().items()
        })

    def clear(self) -> None:
        """Svuota logicamente il buffer in O(1), SENZA resettare il RNG.

        I vecchi valori restano nei buffer allocati ma non sono campionabili.
        Per riavviare anche il RNG, costruire un nuovo ReplayBuffer oppure
        ripristinare lo stato desiderato con load_state_dict().
        """
        self._size = 0
        self._position = 0
        self._total_added = 0

    def state_dict(self) -> dict[str, Any]:
        """Snapshot autonomo delle celle valide, metadati e RNG NumPy.

        Preparato per i checkpoint della Fase 5. I valori NumPy sono copie,
        quindi il chiamante puo' salvarli senza alias dello storage interno.
        """
        state: dict[str, Any] = {
            "version": REPLAY_STATE_VERSION,
            "capacity": self.capacity,
            "size": self._size,
            "position": self._position,
            "total_added": self._total_added,
            "rng_state": deepcopy(self._rng.bit_generator.state),
        }
        state.update({
            key: value[:self._size].copy()
            for key, value in self._arrays().items()
        })
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Ripristina il buffer e la sequenza RNG da uno snapshot coerente.

        Richiede la medesima capacity del buffer costruito. Valida tutto
        prima di sostituire dati e metadati. Non caricare checkpoint non
        attendibili con meccanismi pickle arbitrari.
        """
        if not isinstance(state, dict):
            raise TypeError("state deve essere un dict")
        header = ("version", "capacity", "size", "position", "total_added", "rng_state")
        expected_keys = set(header) | set(self._arrays())
        if set(state) != expected_keys:
            raise ValueError("Chiavi dello state_dict mancanti o inattese")
        if type(state["version"]) is not int or state["version"] != REPLAY_STATE_VERSION:
            raise ValueError("Versione del replay checkpoint incompatibile")
        if type(state["capacity"]) is not int or state["capacity"] != self.capacity:
            raise ValueError("La capacity del replay checkpoint non coincide")
        for name in ("size", "position", "total_added"):
            if type(state[name]) is not int:
                raise TypeError(f"{name} deve essere intero")
        size = state["size"]
        position = state["position"]
        total_added = state["total_added"]
        if not (0 <= size <= self.capacity and 0 <= position < self.capacity
                and total_added >= 0 and size == min(total_added, self.capacity)
                and position == total_added % self.capacity):
            raise ValueError("Metadati size/position/total_added incoerenti")

        new_arrays: dict[str, np.ndarray] = {}
        for name, storage in self._arrays().items():
            array = state[name]
            if (not isinstance(array, np.ndarray) or
                    array.shape != (size, *storage.shape[1:]) or
                    array.dtype != storage.dtype):
                raise ValueError(f"Array checkpoint '{name}' con shape/dtype errati")
            new_arrays[name] = np.ascontiguousarray(array).copy()

        if (not np.isfinite(new_arrays["spatial"]).all() or
                not np.isfinite(new_arrays["features"]).all() or
                not np.isfinite(new_arrays["next_spatial"]).all() or
                not np.isfinite(new_arrays["next_features"]).all() or
                not np.isfinite(new_arrays["rewards"]).all()):
            raise ValueError("Checkpoint contiene valori non finiti")
        for i in range(size):
            _validate_action_mask(new_arrays["action_masks"][i])
            a = int(new_arrays["actions"][i])
            if not 0 <= a < NUM_ACTIONS or not new_arrays["action_masks"][i, a]:
                raise ValueError(f"Azione illegale nel checkpoint all'indice {i}")
            if new_arrays["dones"][i]:
                if (new_arrays["next_action_masks"][i].any() or
                        new_arrays["next_spatial"][i].any() or
                        new_arrays["next_features"][i].any()):
                    raise ValueError("Segnaposto terminale non nullo nel checkpoint")
            else:
                _validate_action_mask(new_arrays["next_action_masks"][i])

        new_rng = np.random.default_rng()
        try:
            new_rng.bit_generator.state = deepcopy(state["rng_state"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("Stato RNG nel checkpoint non valido") from exc

        # Commit dopo tutte le verifiche (gli array snapshot sono gia' isolati).
        for name, storage in self._arrays().items():
            storage[:size] = new_arrays[name]
        self._rng = new_rng
        self._size = size
        self._position = position
        self._total_added = total_added


def to_torch_replay_batch(
    batch: ReplayBatch,
    device: str | torch.device = "cpu",
    *,
    pin_memory: bool = False,
    non_blocking: bool = False,
) -> TorchReplayBatch:
    """Trasforma un minibatch CPU in tensor PyTorch sul device selezionato.

    Tutti i campi vengono trasferiti in un'unica chiamata applicativa.
    Quando CUDA e' disponibile, la combinazione device='cuda',
    pin_memory=True e non_blocking=True consente copie host-to-device
    asincrone; il beneficio va misurato sui minibatch effettivi.

    Dtypes invariati: osservazioni/rewards float32, actions torch.int64,
    maschere/dones torch.bool. Il buffer originale resta in RAM CPU.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch e' necessario per il trasferimento del minibatch") from exc
    if not isinstance(batch, ReplayBatch):
        raise TypeError("batch deve essere un ReplayBatch")
    if pin_memory and not torch.cuda.is_available():
        raise RuntimeError("pin_memory=True richiede CUDA disponibile")

    converted: dict[str, torch.Tensor] = {}
    for name in ReplayBatch.__dataclass_fields__:
        value = getattr(batch, name)
        cpu_tensor = torch.from_numpy(np.ascontiguousarray(value))
        if pin_memory:
            cpu_tensor = cpu_tensor.pin_memory()
        converted[name] = cpu_tensor.to(device=device, non_blocking=non_blocking)
    return TorchReplayBatch(**converted)


__all__ = [
    "ReplayBuffer", "ReplayBatch", "TorchReplayBatch", "to_torch_replay_batch",
    "REPLAY_STATE_VERSION",
]
