"""Seed riproducibili per esperimenti, worker e GridGame.

La coppia (experiment_id, episode_id) identifica un episodio a prescindere
che venga eseguito dal worker 0, 1 o 7. Per questo motivo il seed episodio
NON dipende da worker_id. ``worker_seed`` serve invece per RNG ausiliari
specifici di ciascun processo (non per determinare i seed degli episodi).

GridGame deriva gia' dal seed episodio sei RNG locali (contesto, iniziativa,
obiettivi, Agent, Opponent e combattimento): non duplicarli qui e non
resettarli a ogni attivazione.

``seed_everything`` imposta anche gli RNG globali dove necessario, ma NON
basta a garantire il determinismo assoluto di ogni operazione CUDA e del
multiprocessing asincrono. Registrare sempre gli ID degli episodi, i seed,
la configurazione software e, per un resume esatto, gli stati RNG.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from random import Random
import os
import random
from typing import TypeAlias

import numpy as np


MAX_SEED = (1 << 64) - 1
SeedPart: TypeAlias = int | str


def _seed_uint64(name: str, value: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} deve essere un int (non bool)")
    if not 0 <= value <= MAX_SEED:
        raise ValueError(f"{name} deve essere in [0, 2**64 - 1]")
    return value


def derive_seed(master_seed: int, *parts: SeedPart) -> int:
    """Genera un seed stabile a 64 bit, con separazione di dominio e tipo.

    La sequenza (1, '2') e' distinta da ('1', 2). I nomi di dominio
    (e.g. 'worker', 'episode') separano stream di tipo diverso.
    """
    base = _seed_uint64("master_seed", master_seed)
    digest = blake2b(digest_size=8, person=b"RLGridSeed-v1")
    digest.update(base.to_bytes(8, "big"))
    digest.update(len(parts).to_bytes(4, "big"))
    for i, part in enumerate(parts):
        if type(part) is int:
            raw = _seed_uint64(f"parts[{i}]", part).to_bytes(8, "big")
            tag = b"I"
        elif type(part) is str:
            raw = part.encode("utf-8")
            tag = b"S"
        else:
            raise TypeError(f"parts[{i}] deve essere int o str")
        digest.update(tag)
        digest.update(len(raw).to_bytes(4, "big"))
        digest.update(raw)
    return int.from_bytes(digest.digest(), "big")


@dataclass(frozen=True, slots=True)
class SeedPlan:
    """Piano riproducibile per un esperimento e i suoi processi/episodi."""

    master_seed: int
    experiment_id: int = 0

    def __post_init__(self) -> None:
        _seed_uint64("master_seed", self.master_seed)
        _seed_uint64("experiment_id", self.experiment_id)

    @property
    def experiment_seed(self) -> int:
        return derive_seed(self.master_seed, "experiment", self.experiment_id)

    def worker_seed(self, worker_id: int) -> int:
        return derive_seed(self.experiment_seed, "worker", _seed_uint64("worker_id", worker_id))

    def environment_seed(self, environment_id: int) -> int:
        return derive_seed(self.experiment_seed, "environment", _seed_uint64("environment_id", environment_id))

    def episode_seed(self, episode_id: int) -> int:
        """Dipende dall'ID globale dell'episodio, NON dal numero di worker."""
        return derive_seed(self.experiment_seed, "episode", _seed_uint64("episode_id", episode_id))

    def policy_seed(self, sampler_id: int = 0) -> int:
        """Stream per il campionamento della policy nel processo principale."""
        return derive_seed(self.experiment_seed, "policy", _seed_uint64("sampler_id", sampler_id))

    def shuffle_seed(self) -> int:
        """Seed per ``torch.Generator(device='cpu')`` usato da PPOTrainer."""
        return derive_seed(self.experiment_seed, "ppo-minibatch-shuffle")


def make_python_rng(seed: int) -> Random:
    """RNG locale; non modifica ``random`` globale."""
    return Random(_seed_uint64("seed", seed))


def make_numpy_rng(seed: int) -> np.random.Generator:
    """RNG locale NumPy (Generator/PCG64), indipendente dall'RNG globale."""
    return np.random.default_rng(_seed_uint64("seed", seed))


def make_torch_generator(seed: int, *, device: str = "cpu"):
    """Crea un ``torch.Generator`` locale su CPU o CUDA.

    Un generatore CUDA va creato SOLO nel processo principale, mai nei
    worker ``spawn`` che simulano GridGame su CPU.
    """
    _seed_uint64("seed", seed)
    if not isinstance(device, str):
        raise TypeError("device deve essere una stringa, es. 'cpu' o 'cuda:0'")
    import torch  # import ritardato: i worker CPU non richiedono PyTorch

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def seed_everything(
    seed: int,
    *,
    seed_torch: bool = True,
    seed_cuda: bool = False,
    deterministic_algorithms: bool = False,
) -> None:
    """Imposta RNG globali Python, NumPy e (opzionalmente) PyTorch.

    ``seed_cuda=False`` di default per sicurezza nei processi worker CPU.
    Con ``deterministic_algorithms=True`` occorre chiamare questa funzione
    PRIMA di inizializzare CUDA; alcune operazioni possono non avere un
    kernel deterministico. Il determinismo rigoroso non e' garantito fra
    differenti piattaforme/versioni di PyTorch.

    Per generare singoli episodi preferire ``SeedPlan.episode_seed`` e
    ``GridGame.reset(seed=...)`` al ripetuto reseeding dei globali.
    """
    _seed_uint64("seed", seed)
    for name, flag in (
        ("seed_torch", seed_torch),
        ("seed_cuda", seed_cuda),
        ("deterministic_algorithms", deterministic_algorithms),
    ):
        if type(flag) is not bool:
            raise TypeError(f"{name} deve essere bool")
    if (seed_cuda or deterministic_algorithms) and not seed_torch:
        raise ValueError("seed_cuda/deterministic_algorithms richiedono seed_torch=True")

    random.seed(derive_seed(seed, "python-global"))
    np.random.seed(derive_seed(seed, "numpy-legacy-global") % (1 << 32))

    if seed_torch:
        if deterministic_algorithms:
            # Deve essere impostata prima della prima inizializzazione CUDA.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        # A differenza di torch.manual_seed(), agisce esclusivamente sul
        # default generator CPU e non programma chiamate CUDA nei worker.
        torch.random.default_generator.manual_seed(derive_seed(seed, "torch-cpu-global"))
        if seed_cuda:
            torch.cuda.manual_seed_all(derive_seed(seed, "torch-cuda-global"))
        if deterministic_algorithms:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


__all__ = [
    "MAX_SEED", "SeedPlan", "derive_seed", "make_python_rng",
    "make_numpy_rng", "make_torch_generator", "seed_everything",
]
