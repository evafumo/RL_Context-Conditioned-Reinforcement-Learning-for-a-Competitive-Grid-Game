"""DQN standard (NON Double DQN) per Giulio2: 24 contesti, action masking.
MDP:
    Il simulatore ha 100 attivazioni, incluse le attivazioni forzate DUMMY.
    Il gamma ufficiale del progetto e' 1. Il terminale s_dagger non e' uno
    stato osservabile: il replay vi associa done=True, next_* tutti zero.

Target Bellman, con *soltanto* la target network per il massimo:
    y = r                                      se done
    y = r + gamma * max_{a' legale} Q_target(s',a') altrimenti.

Q_online(s,a) e' estratto dai sette Q-value GREZZI tramite gather():
le maschere NON vengono applicate al Q della particolare azione eseguita.
L'unico masking del target usa environment.action_mask.mask_q_values().

La validazione normativa degli input (azioni legali e maschere) e' affidata
al ReplayBuffer. Si controllano sempre shape/dtype/device dei tensori; su
CPU si controllano anche le invarianti del contenuto. Su CUDA queste ultime
sono affidate al ReplayBuffer, per evitare sincronizzazioni GPU-CPU ad ogni
update. Non si ricostruiscono in questo modulo le regole del gioco.

Gli ambienti restano su CPU, replay in RAM CPU, la rete e i minibatch sono
spostati sul device del trainer. CUDA NON viene attivata all'importazione.
Float32 e' il default; autocast BF16 su GPU e' facoltativo e non richiede
GradScaler. Il beneficio di pin_memory/non_blocking e BF16 va misurato.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Any, Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from environment.action_mask import NUM_ACTIONS, mask_q_values
from environment.observations import FEATURE_SIZE, SPATIAL_SHAPE
from environment.rules import Action
from agent.network import QNetwork
from agent.replay_buffer import (
    ReplayBatch,
    ReplayBuffer,
    TorchReplayBatch,
    to_torch_replay_batch,
)

DQN_TRAINER_STATE_VERSION: Final[int] = 1


def _positive_real(value: float, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} deve essere un numero reale, non un booleano")
    number = float(value)
    if not math.isfinite(number) or (number < 0 if allow_zero else number <= 0):
        raise ValueError(f"{name} deve essere finito e {'non negativo' if allow_zero else 'positivo'}")
    return number


def _validate_gamma(gamma: float) -> float:
    if isinstance(gamma, bool) or not isinstance(gamma, Real):
        raise TypeError("gamma deve essere un numero reale")
    value = float(gamma)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("gamma deve appartenere a [0,1]")
    return value


@dataclass(frozen=True, slots=True)
class DQNConfig:
    """Iperparametri del singolo aggiornamento DQN (schedule in train.py).

    gamma=1 e' la scelta prevista dal progetto. Le frequenze sono espresse
    in NUMERO DI UPDATE DEL GRADIENTE, non in attivazioni dell'ambiente.
    """

    gamma: float = 1.0
    learning_rate: float = 1e-4
    target_sync_interval: int = 1_000
    huber_delta: float = 1.0
    grad_clip_norm: float = 10.0
    weight_decay: float = 0.0
    use_bfloat16_amp: bool = False

    def __post_init__(self) -> None:
        _validate_gamma(self.gamma)
        _positive_real(self.learning_rate, "learning_rate")
        _positive_real(self.huber_delta, "huber_delta")
        _positive_real(self.grad_clip_norm, "grad_clip_norm")
        _positive_real(self.weight_decay, "weight_decay", allow_zero=True)
        if type(self.target_sync_interval) is not int or self.target_sync_interval <= 0:
            raise ValueError("target_sync_interval deve essere un intero > 0")
        if type(self.use_bfloat16_amp) is not bool:
            raise TypeError("use_bfloat16_amp deve essere booleano")


@dataclass(frozen=True, slots=True)
class DQNUpdateStats:
    """Statistiche di un update, scalari Python per il logging della Fase 5."""

    loss: float
    mean_q: float
    mean_target: float
    grad_norm: float  # Norma L2 PRIMA del gradient clipping.
    update_steps: int
    target_synced: bool
    td_mae: float                  # Media |target Bellman - Q(s,a)|.
    td_p90: float                  # 90o percentile |TD error| (quantile lineare).
    std_q: float                   # Deviazione standard di popolazione di Q(s,a).
    mean_max_legal_q: float        # Media max_{a legale} Q_online(s,a).
    std_max_legal_q: float         # Deviazione standard di popolazione del max legale.


def _validate_batch(batch: TorchReplayBatch) -> None:
    """Controlli strutturali (GPU senza sync) e controlli normativi su CPU.

    Un batch CUDA deriva normalmente da ReplayBuffer.add()/sample() che
    verifica gia' i valori della maschera e la legalita' dell'azione.
    """
    if not isinstance(batch, TorchReplayBatch):
        raise TypeError("batch deve essere un TorchReplayBatch")
    b = batch.batch_size
    if b <= 0:
        raise ValueError("Il minibatch non puo' essere vuoto")
    specs = {
        "spatial": ((b, *SPATIAL_SHAPE), torch.float32),
        "features": ((b, FEATURE_SIZE), torch.float32),
        "action_masks": ((b, NUM_ACTIONS), torch.bool),
        "actions": ((b,), torch.int64),
        "rewards": ((b,), torch.float32),
        "next_spatial": ((b, *SPATIAL_SHAPE), torch.float32),
        "next_features": ((b, FEATURE_SIZE), torch.float32),
        "next_action_masks": ((b, NUM_ACTIONS), torch.bool),
        "dones": ((b,), torch.bool),
    }
    device = batch.actions.device
    for name, (shape, dtype) in specs.items():
        value = getattr(batch, name)
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"batch.{name} deve avere shape {shape}")
        if value.dtype != dtype:
            raise TypeError(f"batch.{name} deve avere dtype {dtype}")
        if value.device != device:
            raise ValueError("Tutti i campi del minibatch devono stare sullo stesso device")

    if device.type == "cpu":
        # Su CPU il controllo dei dati evita il costo di sincronizzazione CUDA.
        if not all(torch.isfinite(getattr(batch, key)).all().item() for key in (
            "spatial", "features", "rewards", "next_spatial", "next_features"
        )):
            raise ValueError("Il minibatch contiene valori non finiti")
        if not bool(((batch.actions >= 0) & (batch.actions < NUM_ACTIONS)).all()):
            raise ValueError("ID azione fuori dall'intervallo 0..6")
        if not bool(batch.action_masks.gather(1, batch.actions[:, None]).all()):
            raise ValueError("Nel minibatch e' presente un'azione illegale")
        dummy = batch.action_masks[:, int(Action.DUMMY)]
        valid_current = (
            batch.action_masks.any(dim=1)
            & ((~dummy) | (batch.action_masks.sum(dim=1) == 1))
            & (dummy | batch.action_masks[:, int(Action.PASS)])
        )
        if not bool(valid_current.all()):
            raise ValueError("Una maschera corrente viola le invarianti PASS/DUMMY")
        nonterminal = ~batch.dones
        if bool((nonterminal & ~batch.next_action_masks.any(dim=1)).any()):
            raise ValueError("Uno stato non terminale richiede una maschera successiva non vuota")
        next_dummy = batch.next_action_masks[:, int(Action.DUMMY)]
        invalid_next = (
            nonterminal & (
                (next_dummy & (batch.next_action_masks.sum(dim=1) != 1))
                | (~next_dummy & ~batch.next_action_masks[:, int(Action.PASS)])
            )
        )
        if bool(invalid_next.any()):
            raise ValueError("La maschera successiva viola PASS/DUMMY")
        if bool((batch.dones & batch.next_action_masks.any(dim=1)).any()):
            raise ValueError("La maschera successiva terminale deve essere tutta False")
        if bool(batch.dones.any()) and (bool(batch.next_spatial[batch.dones].any())
                                        or bool(batch.next_features[batch.dones].any())):
            raise ValueError("Gli stati successivi terminali devono essere segnaposto zero")


def _validate_q_shape(q_values: Tensor, batch_size: int) -> None:
    if not isinstance(q_values, Tensor) or q_values.shape != (batch_size, NUM_ACTIONS):
        raise ValueError(f"La rete deve restituire Q-value (B,{NUM_ACTIONS})")
    if not q_values.is_floating_point():
        raise TypeError("I Q-value devono essere floating point")


@torch.no_grad()
def compute_dqn_targets(
    target_network: nn.Module,
    batch: TorchReplayBatch,
    *,
    gamma: float = 1.0,
) -> Tensor:
    """Calcola Bellman target (B,) senza gradienti e senza Double DQN.

    Il massimo e' preso ESCLUSIVAMENTE da target_network, applicando
    mask_q_values() della Fase 2 alle azioni legali di s'. Per done=True
    sono assenti azioni legali: si rende temporaneamente la maschera tutta
    True, poi torch.where() azzera il bootstrap prima di sommare il reward.
    In questo modo non si calcola mai 0 * (-inf), che sarebbe NaN.
    """
    if not isinstance(target_network, nn.Module):
        raise TypeError("target_network deve essere un nn.Module")
    discount = _validate_gamma(gamma)
    _validate_batch(batch)
    next_q = target_network(batch.next_spatial, batch.next_features)
    _validate_q_shape(next_q, batch.batch_size)
    if next_q.device != batch.rewards.device:
        raise ValueError("Target network e batch devono stare sullo stesso device")

    # Le righe done non possiedono un insieme di azioni legali: maschera
    # provvisoria tutta True SOLO per evitare un max di tutti -inf.
    safe_masks = batch.next_action_masks | batch.dones.unsqueeze(1)
    legal_next_q = mask_q_values(next_q, safe_masks)
    best_next_q = legal_next_q.max(dim=1).values
    bootstrap = torch.where(batch.dones, torch.zeros_like(best_next_q), best_next_q)
    return batch.rewards + discount * bootstrap


def _compute_dqn_loss_components(
    online_network: nn.Module,
    target_network: nn.Module,
    batch: TorchReplayBatch,
    *,
    gamma: float = 1.0,
    huber_delta: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Loss, Q(s,a), target y e matrice Q_online(s,:) della stessa forward.

    La matrice completa serve soltanto alle statistiche diagnostiche.
    Non viene applicato alcun masking a Q(s,a) nella loss e il target
    Bellman resta calcolato unicamente con la target network.
    """
    if not isinstance(online_network, nn.Module):
        raise TypeError("online_network deve essere un nn.Module")
    _positive_real(huber_delta, "huber_delta")
    _validate_batch(batch)
    all_online_q = online_network(batch.spatial, batch.features)
    _validate_q_shape(all_online_q, batch.batch_size)
    if all_online_q.device != batch.actions.device:
        raise ValueError("Rete online e batch devono stare sullo stesso device")
    chosen_q = all_online_q.gather(dim=1, index=batch.actions.unsqueeze(1)).squeeze(1)
    targets = compute_dqn_targets(target_network, batch, gamma=gamma)
    loss = F.smooth_l1_loss(chosen_q.float(), targets.float(), beta=huber_delta)
    return loss, chosen_q, targets, all_online_q


def compute_dqn_loss(
    online_network: nn.Module,
    target_network: nn.Module,
    batch: TorchReplayBatch,
    *,
    gamma: float = 1.0,
    huber_delta: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Restituisce (loss scalare, Q_online(s,a), target y), senza step().

    Firma e contratto invariati. La loss e' Huber media; i gradienti
    attraversano soltanto Q_online(s,a), mai la target network.
    """
    loss, chosen_q, targets, _ = _compute_dqn_loss_components(
        online_network, target_network, batch,
        gamma=gamma, huber_delta=huber_delta,
    )
    return loss, chosen_q, targets


@torch.no_grad()
def hard_sync_target(online_network: nn.Module, target_network: nn.Module) -> None:
    """Copia tutti i pesi e i buffer online -> target, senza soft update."""
    if not isinstance(online_network, nn.Module) or not isinstance(target_network, nn.Module):
        raise TypeError("Entrambe le reti devono essere nn.Module")
    target_network.load_state_dict(online_network.state_dict(), strict=True)
    target_network.eval()
    target_network.requires_grad_(False)


class DQNTrainer:
    """Unita' di ottimizzazione DQN con online/target network indipendenti.

    Args:
        online_network: QNetwork (default) o modulo nn.Module compatibile.
        config: iperparametri, immutabili.
        device: 'cpu', 'cuda', 'cuda:0' o torch.device.

    L'esplorazione epsilon-greedy, il collector, i reset, gli orizzonti,
    il numero di update per transizione e i checkpoint su disco NON sono
    responsabilita' di questa classe.
    """

    def __init__(
        self,
        online_network: nn.Module | None = None,
        *,
        config: DQNConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if config is None:
            config = DQNConfig()
        if not isinstance(config, DQNConfig):
            raise TypeError("config deve essere DQNConfig")
        if online_network is None:
            online_network = QNetwork()
        if not isinstance(online_network, nn.Module):
            raise TypeError("online_network deve essere nn.Module")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA non disponibile: controllare installazione PyTorch CUDA")
        if config.use_bfloat16_amp:
            if self.device.type != "cuda":
                raise ValueError("BF16 autocast e' supportato qui soltanto su CUDA")
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("La GPU/PyTorch corrente non supporta BF16")
        self.config = config
        self.online_network = online_network.to(self.device)
        self.target_network = deepcopy(self.online_network).to(self.device)
        hard_sync_target(self.online_network, self.target_network)
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.update_steps = 0

    def _on_device(self, batch: ReplayBatch | TorchReplayBatch) -> TorchReplayBatch:
        if isinstance(batch, ReplayBatch):
            return to_torch_replay_batch(
                batch,
                device=self.device,
                pin_memory=self.device.type == "cuda",
                non_blocking=self.device.type == "cuda",
            )
        if not isinstance(batch, TorchReplayBatch):
            raise TypeError("batch deve essere ReplayBatch o TorchReplayBatch")
        if batch.actions.device != self.device:
            raise ValueError(
                "TorchReplayBatch sul device errato: convertirlo con "
                "to_torch_replay_batch(batch, device=...)"
            )
        return batch

    def train_step(self, batch: ReplayBatch | TorchReplayBatch) -> DQNUpdateStats:
        """Un singolo update Adam + eventuale hard sync della target network.

        Non seleziona le azioni e non campiona il replay. In CUDA, un
        ReplayBatch NumPy viene trasferito in blocco con pin_memory.
        Restituisce statistiche come scalari per il logging (sincronizzazione
        GPU necessaria: raggruppata in un solo trasferimento finale).
        """
        torch_batch = self._on_device(batch)
        self.online_network.train()
        self.target_network.eval()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16 if self.config.use_bfloat16_amp else torch.float32,
            enabled=self.config.use_bfloat16_amp,
        ):
            loss, q_values, targets, all_online_q = _compute_dqn_loss_components(
                self.online_network, self.target_network, torch_batch,
                gamma=self.config.gamma, huber_delta=self.config.huber_delta,
            )
        # Diagnostiche calcolate PRIMA dell'optimizer.step(): anche una rete
        # con output view dei propri parametri (es. TableQ dei test) produce
        # statistiche coerenti con i Q usati dalla loss di questo update.
        # L'operazione e' separata dall'autograd e non cambia il Bellman target.
        with torch.no_grad():
            selected_q = q_values.detach().float()
            bellman_targets = targets.detach().float()
            abs_td = (bellman_targets - selected_q).abs()
            legal_max_q = mask_q_values(
                all_online_q.detach().float(), torch_batch.action_masks,
            ).max(dim=1).values
            diagnostic_values = torch.stack((
                loss.detach().float(),
                selected_q.mean(),
                bellman_targets.mean(),
                abs_td.mean(),
                torch.quantile(abs_td, 0.90),
                selected_q.std(unbiased=False),
                legal_max_q.mean(),
                legal_max_q.std(unbiased=False),
            ))
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.online_network.parameters(), self.config.grad_clip_norm,
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.update_steps += 1
        synced = self.update_steps % self.config.target_sync_interval == 0
        if synced:
            self.sync_target()
        # Un solo trasferimento GPU->CPU, nove scalari per update.
        values = torch.cat((
            diagnostic_values, grad_norm.detach().float().reshape(1),
        )).cpu().tolist()
        return DQNUpdateStats(
            loss=values[0], mean_q=values[1], mean_target=values[2],
            grad_norm=values[8], update_steps=self.update_steps,
            target_synced=synced, td_mae=values[3], td_p90=values[4],
            std_q=values[5], mean_max_legal_q=values[6],
            std_max_legal_q=values[7],
        )

    def train_on_replay(self, replay: ReplayBuffer, batch_size: int) -> DQNUpdateStats:
        """Estrae un minibatch uniforme senza rimpiazzo e applica train_step()."""
        if not isinstance(replay, ReplayBuffer):
            raise TypeError("replay deve essere ReplayBuffer")
        return self.train_step(replay.sample(batch_size))

    def sync_target(self) -> None:
        """Sincronizzazione hard esplicita, indipendente dal contatore update."""
        hard_sync_target(self.online_network, self.target_network)

    def state_dict(self) -> dict[str, Any]:
        """Stato completo riavviabile; salvataggio su disco in utils/checkpoint.py."""
        return {
            "version": DQN_TRAINER_STATE_VERSION,
            "config": asdict(self.config),
            "online_network": deepcopy(self.online_network.state_dict()),
            "target_network": deepcopy(self.target_network.state_dict()),
            "optimizer": deepcopy(self.optimizer.state_dict()),
            "update_steps": self.update_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Ripristina modelli, ottimizzatore e conteggio update.

        Occorre istanziare il trainer con la stessa configurazione. La
        scelta del device e' quella del trainer corrente (CPU/CUDA).
        Caricare soltanto checkpoint da fonti fidate.
        """
        if not isinstance(state, dict):
            raise TypeError("state deve essere un dizionario")
        if set(state) != {
            "version", "config", "online_network", "target_network", "optimizer", "update_steps"
        } or state["version"] != DQN_TRAINER_STATE_VERSION:
            raise ValueError("Formato/versione del checkpoint DQN non valido")
        if state["config"] != asdict(self.config):
            raise ValueError("Configurazione del trainer incompatibile con il checkpoint")
        steps = state["update_steps"]
        if type(steps) is not int or steps < 0:
            raise ValueError("update_steps non valido")
        self.online_network.load_state_dict(state["online_network"], strict=True)
        self.target_network.load_state_dict(state["target_network"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.target_network.eval()
        self.target_network.requires_grad_(False)
        self.update_steps = steps


__all__ = [
    "DQNConfig", "DQNUpdateStats", "DQNTrainer", "compute_dqn_targets",
    "compute_dqn_loss", "hard_sync_target", "DQN_TRAINER_STATE_VERSION",
]
