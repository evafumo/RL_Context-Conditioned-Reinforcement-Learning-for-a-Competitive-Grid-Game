"""Benchmark end-to-end del Context-Conditioned Masked DQN standard.

Esecuzione dalla radice::

    python -m benchmarks.benchmark_dqn --device cuda:0 --envs 8 \
        --workers 8 4 2 1 --simulation-steps 100 \
        --steps-per-env 100 --inference-iters 100 \
        --updates 32 --cycle-updates 32 --batch-size 256 --pin-memory

Cosa viene misurato (tutti i tempi escludono l'avvio dei worker):
    simulation: solo ParallelGridGames.step con azioni legali CPU.
    inference: conversione CPU->GPU, selezione epsilon-greedy, copia GPU->CPU;
               il batch e' riutilizzato, NON viene eseguito alcun env.step.
    collection: collect_parallel_experience, incluso il replay su RAM CPU.
    updates: replay.sample + DQNTrainer.train_step (loss, backward, Adam).
    cycle: collect_parallel_experience + N aggiornamenti DQN reali.

Il benchmark NON cambia reward, maschere, terminali, DUMMY o Bellman targets.
La CPU dei worker NON esegue modelli PyTorch: import pesanti ritardati, per
non inizializzare CUDA nei processi figli spawn su Windows.

I risultati dipendono da sistema operativo, GPU, CPU, configurazione Torch,
contesa di risorse e temperatura hardware. Ripetere piu' volte per confronti
statisticamente robusti. Le statistiche degli update includono il costo del
logging degli scalari, previsto anche dal DQNTrainer reale.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from math import isfinite
import os
from pathlib import Path
from time import perf_counter
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from agent.parallel_env import ParallelGridGames, ParallelObservation
    from agent.replay_buffer import ReplayBuffer
    from agent.dqn import DQNTrainer


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    device: str = "cpu"
    envs: int = 8
    workers: tuple[int, ...] = (8, 4, 2, 1)
    simulation_steps: int = 100
    steps_per_env: int = 100
    inference_iters: int = 100
    updates: int = 32
    cycle_steps_per_env: int = 100
    cycle_updates: int = 32
    batch_size: int = 256
    replay_capacity: int = 20_000
    epsilon: float = 0.1
    seed: int = 20261026
    experiment_id: int = 0
    shaping_lambda: float = 0.1
    pin_memory: bool = False
    bf16: bool = False
    warmup_simulation_steps: int = 5
    warmup_inference_iters: int = 10
    warmup_collection_steps: int = 2
    warmup_updates: int = 3
    torch_threads: int = 1
    output: str | Path | None = "benchmarks/results/benchmark_dqn.json"

    def __post_init__(self) -> None:
        import torch

        positive = ("envs", "simulation_steps", "steps_per_env", "inference_iters",
                    "updates", "cycle_steps_per_env", "cycle_updates", "batch_size",
                    "replay_capacity", "torch_threads")
        nonnegative = ("seed", "experiment_id", "warmup_simulation_steps",
                       "warmup_inference_iters", "warmup_collection_steps", "warmup_updates")
        for field in positive:
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} deve essere un intero positivo (non bool)")
        for field in nonnegative:
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} deve essere un intero >= 0 (non bool)")
        if self.seed >= 2**64 or self.experiment_id >= 2**64:
            raise ValueError("seed ed experiment_id devono essere uint64")
        if (not isinstance(self.workers, tuple) or not self.workers
                or any(type(w) is not int or w < 1 or w > self.envs for w in self.workers)
                or len(set(self.workers)) != len(self.workers)):
            raise ValueError("workers: worker distinti e compresi tra 1 ed envs")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity deve essere >= batch_size")
        for field in ("pin_memory", "bf16"):
            if type(getattr(self, field)) is not bool:
                raise TypeError(f"{field} deve essere bool")
        for field in ("epsilon", "shaping_lambda"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field} deve essere numerico")
            if not isfinite(value) or (not 0 <= value <= 1 if field == "epsilon" else value < 0):
                raise ValueError(f"{field} fuori intervallo")
        if not isinstance(self.device, str):
            raise TypeError("device deve essere una stringa")
        try:
            parsed_device = torch.device(self.device)
        except (RuntimeError, ValueError, TypeError) as exc:
            raise ValueError("device PyTorch non valido") from exc
        if parsed_device.type not in ("cpu", "cuda"):
            raise ValueError("Sono supportati solo cpu e cuda[:indice]")
        if parsed_device.type != "cuda" and (self.pin_memory or self.bf16):
            raise ValueError("pin_memory e bf16 richiedono CUDA")
        if self.output is not None and not str(self.output).strip():
            raise ValueError("output non puo' essere vuoto")


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    workers: int
    envs: int
    worker_startup_seconds: float
    simulation: dict[str, Any]
    inference: dict[str, Any]
    collection: dict[str, Any]
    updates: dict[str, Any]
    cycle: dict[str, Any]
    replay_bytes: int
    gpu_peak_allocated_bytes: int | None
    gpu_peak_reserved_bytes: int | None


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    config: BenchmarkConfig
    hardware: dict[str, Any]
    results: tuple[BenchmarkResult, ...]


def _sync(device: "torch.device") -> None:
    """Le operazioni CUDA sono asincrone: sincronizza solo nel coordinatore."""
    if device.type == "cuda":
        import torch
        torch.cuda.synchronize(device)


def _rate(count: int, seconds: float) -> float:
    from utils.metrics import throughput
    return throughput(count, max(seconds, 1e-12))


def _expected_terminals(initial_activations: Any, steps: int) -> int:
    """Numero di step terminali senza interpretare mappe/avversari."""
    from environment.game import HORIZON
    return sum((int(a) - 1 + steps) // HORIZON for a in initial_activations)


def _validate_collection(stats: Any, envs: "ParallelGridGames", *,
                         steps: int, expected_terminal: int, replay: "ReplayBuffer",
                         before_added: int) -> None:
    from agent.collector import CollectionStats
    expected = envs.num_envs * steps
    if not isinstance(stats, CollectionStats):
        raise RuntimeError("Il collector non ha prodotto CollectionStats")
    if (stats.transitions != expected or stats.steps_per_env != steps
            or stats.num_envs != envs.num_envs
            or stats.agent_decisions + stats.forced_activations != expected
            or stats.completed_episodes != expected_terminal
            or not 0 <= stats.wins <= stats.completed_episodes
            or not isfinite(stats.rewards_sum)
            or replay.total_added - before_added != expected):
        raise RuntimeError("Statistiche collector / replay non compatibili con l'MDP")


def benchmark_simulation(environments: "ParallelGridGames", *, steps: int) -> dict[str, Any]:
    """Solo simulatore parallelo: un'azione legale per ogni attivazione MDP."""
    import numpy as np

    if type(steps) is not int or steps <= 0:
        raise ValueError("steps deve essere > 0")
    initial = environments.observation.activations.copy()
    initial_indices = environments.episode_indices
    count = dones = forced = 0
    reward_sum = 0.0
    start = perf_counter()
    for _ in range(steps):
        obs = environments.observation
        # Maschera autorevole gia' calcolata dai worker; DUMMY se forzato.
        actions = obs.action_masks.argmax(axis=1).astype(np.int64)
        result = environments.step(actions)
        dones += int(np.count_nonzero(result.dones))
        forced += int(np.count_nonzero(~obs.is_agent_decision))
        count += environments.num_envs
        reward_sum += float(np.sum(result.rewards, dtype=np.float64))
    elapsed = max(perf_counter() - start, 1e-12)
    expected = _expected_terminals(initial, steps)
    if (dones != expected or count != environments.num_envs * steps
            or sum(b - a for a, b in zip(initial_indices, environments.episode_indices)) != expected):
        raise RuntimeError("Simulazione: orizzonte terminale o contatori errati")
    return {
        "seconds": elapsed, "activations": count,
        "activations_per_second": _rate(count, elapsed),
        "completed_episodes": dones, "forced_activations": forced,
        "rewards_sum": reward_sum,
    }


def benchmark_inference(
    observation: "ParallelObservation", network: Any, *, device: "torch.device",
    iterations: int, warmup_iterations: int = 0, epsilon: float = 0.1,
    generator: "torch.Generator | None" = None, pin_memory: bool = False,
) -> dict[str, Any]:
    """Batch H2D + forward/policy + D2H: ESCLUDE env.step e replay.

    Riutilizza lo stesso snapshot per evitare che la simulazione contamini la
    misura. Include le sincronizzazioni GPU->CPU effettive delle azioni.
    """
    import numpy as np
    import torch
    from agent.action_selection import select_actions
    from environment.observations import ObservationBatch, to_torch_batch
    from environment.action_mask import to_torch_action_masks

    if type(iterations) is not int or iterations <= 0:
        raise ValueError("iterations deve essere positivo")
    if type(warmup_iterations) is not int or warmup_iterations < 0:
        raise ValueError("warmup_iterations deve essere >= 0")
    network.eval()

    def one_batch() -> None:
        batch = to_torch_batch(
            ObservationBatch(observation.spatial, observation.features), device,
            pin_memory=pin_memory, non_blocking=pin_memory,
        )
        masks = to_torch_action_masks(
            observation.action_masks, device, pin_memory=pin_memory,
            non_blocking=pin_memory,
        )
        chosen = select_actions(network, batch.spatial, batch.features, masks,
                                epsilon, generator=generator)
        ids = chosen.cpu().numpy()  # Copia D2H + sincronizzazione necessaria.
        if ids.shape != (observation.num_envs,) or not np.all(
                observation.action_masks[np.arange(observation.num_envs), ids]):
            raise RuntimeError("L'inferenza ha selezionato azioni illegali")

    with torch.inference_mode():
        for _ in range(warmup_iterations):
            one_batch()
        _sync(device)
        start = perf_counter()
        for _ in range(iterations):
            one_batch()
        _sync(device)
        elapsed = max(perf_counter() - start, 1e-12)
    observations_count = iterations * observation.num_envs
    return {
        "seconds": elapsed, "batch_calls": iterations,
        "observations": observations_count,
        "batches_per_second": _rate(iterations, elapsed),
        "observations_per_second": _rate(observations_count, elapsed),
        "milliseconds_per_batch": 1000.0 * elapsed / iterations,
    }


def benchmark_collection(
    environments: "ParallelGridGames", network: Any, replay: "ReplayBuffer", *,
    steps_per_env: int, epsilon: float, device: "torch.device", pin_memory: bool,
    generator: "torch.Generator | None" = None,
) -> dict[str, Any]:
    """Misura il collector REALE con maschere, DUMMY e replay NumPy."""
    from agent.collector import collect_parallel_experience

    before_added = replay.total_added
    initial_activations = environments.observation.activations.copy()
    expected = _expected_terminals(initial_activations, steps_per_env)
    _sync(device)
    start = perf_counter()
    stats = collect_parallel_experience(
        environments, network, replay, steps_per_env=steps_per_env,
        epsilon=epsilon, device=device, pin_memory=pin_memory, generator=generator,
    )
    _sync(device)
    elapsed = max(perf_counter() - start, 1e-12)
    _validate_collection(stats, environments, steps=steps_per_env,
                         expected_terminal=expected, replay=replay, before_added=before_added)
    return {
        "seconds": elapsed, "transitions": stats.transitions,
        "transitions_per_second": _rate(stats.transitions, elapsed),
        "completed_episodes": stats.completed_episodes, "wins": stats.wins,
        "agent_decisions": stats.agent_decisions,
        "forced_activations": stats.forced_activations,
        "rewards_sum": stats.rewards_sum,
    }


def benchmark_updates(
    trainer: "DQNTrainer", replay: "ReplayBuffer", *,
    updates: int, batch_size: int, warmup_updates: int = 0,
) -> dict[str, Any]:
    """Misura sample uniforme + DQNTrainer.train_on_replay (non Double DQN)."""
    if any(type(value) is not int for value in (updates, batch_size, warmup_updates)):
        raise TypeError("updates, batch_size e warmup_updates devono essere int")
    if updates < 1 or batch_size < 1 or warmup_updates < 0:
        raise ValueError("updates/batch_size > 0, warmup_updates >= 0")
    if len(replay) < batch_size:
        raise ValueError("Replay non sufficientemente pieno per un minibatch")
    for _ in range(warmup_updates):
        trainer.train_on_replay(replay, batch_size)
    device = trainer.device
    _sync(device)
    first_step = trainer.update_steps
    start = perf_counter()
    last = None
    for _ in range(updates):
        last = trainer.train_on_replay(replay, batch_size)
    _sync(device)
    elapsed = max(perf_counter() - start, 1e-12)
    if last is None or trainer.update_steps != first_step + updates:
        raise RuntimeError("Conteggio aggiornamenti DQN incoerente")
    if not all(isfinite(value) for value in (last.loss, last.mean_q, last.mean_target, last.grad_norm)):
        raise RuntimeError("Metriche DQN non finite")
    return {
        "seconds": elapsed, "updates": updates, "batch_size": batch_size,
        "updates_per_second": _rate(updates, elapsed),
        "sampled_transitions_per_second": _rate(updates * batch_size, elapsed),
        "last_loss": float(last.loss), "last_mean_q": float(last.mean_q),
        "last_mean_target": float(last.mean_target),
        "last_grad_norm": float(last.grad_norm),
        "target_syncs": ((first_step + updates) // trainer.config.target_sync_interval
                         - first_step // trainer.config.target_sync_interval),
    }


def benchmark_cycle(
    environments: "ParallelGridGames", trainer: "DQNTrainer",
    replay: "ReplayBuffer", *, steps_per_env: int, updates: int,
    batch_size: int, epsilon: float, pin_memory: bool,
    generator: "torch.Generator | None" = None,
) -> dict[str, Any]:
    """Misura assieme collection + N update, senza startup dei worker."""
    from agent.collector import collect_parallel_experience

    device = trainer.device
    start_activations = environments.observation.activations.copy()
    before_added = replay.total_added
    expected_done = _expected_terminals(start_activations, steps_per_env)
    _sync(device)
    start = perf_counter()
    collected = collect_parallel_experience(
        environments, trainer.online_network, replay,
        steps_per_env=steps_per_env, epsilon=epsilon, device=device,
        pin_memory=pin_memory, generator=generator,
    )
    if len(replay) < batch_size:
        raise RuntimeError("Replay troppo piccolo per gli update del ciclo")
    for _ in range(updates):
        trainer.train_on_replay(replay, batch_size)
    _sync(device)
    elapsed = max(perf_counter() - start, 1e-12)
    _validate_collection(collected, environments, steps=steps_per_env,
                         expected_terminal=expected_done, replay=replay,
                         before_added=before_added)
    return {
        "seconds": elapsed, "transitions": collected.transitions,
        "gradient_updates": updates,
        "activations_per_second": _rate(collected.transitions, elapsed),
        "updates_per_second": _rate(updates, elapsed),
        "completed_episodes": collected.completed_episodes,
        "agent_decisions": collected.agent_decisions,
        "forced_activations": collected.forced_activations,
    }


def _resolve_device(requested: str) -> "torch.device":
    import torch
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA non disponibile: verificare build CUDA di PyTorch")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"GPU cuda:{index} non presente")
        return torch.device(f"cuda:{index}")
    return device


def _hardware(device: "torch.device") -> dict[str, Any]:
    import platform
    import torch

    hardware = {
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu_logical_cores": os.cpu_count(), "pytorch": torch.__version__,
        "torch_cuda_build": torch.version.cuda, "device": str(device),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        hardware.update(gpu_name=props.name,
                        gpu_total_memory_bytes=props.total_memory)
    return hardware


def _write_report(report: BenchmarkReport, target: str | Path) -> Path:
    """Scrittura JSON atomica: risultato parziale integro in caso d'interruzione."""
    import tempfile

    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    ready = asdict(report)
    if ready["config"]["output"] is not None:
        ready["config"]["output"] = str(ready["config"]["output"])
    payload = json.dumps(ready, indent=2, ensure_ascii=False,
                         sort_keys=True, allow_nan=False) + "\n"
    name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".dqn-benchmark-",
            suffix=".tmp", delete=False, newline="\n"
        ) as stream:
            name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name is not None and os.path.exists(name):
            os.unlink(name)
    return path


def _prefill_replay(environments: "ParallelGridGames", trainer: "DQNTrainer",
                    replay: "ReplayBuffer", config: BenchmarkConfig,
                    generator: "torch.Generator") -> int:
    """Prefill NON cronometrato, necessario solo per minibatch molto grandi."""
    from agent.collector import collect_parallel_experience

    if len(replay) >= config.batch_size:
        return 0
    missing_steps = (config.batch_size - len(replay) + config.envs - 1) // config.envs
    stats = collect_parallel_experience(
        environments, trainer.online_network, replay,
        steps_per_env=missing_steps, epsilon=config.epsilon,
        device=trainer.device, pin_memory=config.pin_memory, generator=generator,
    )
    if len(replay) < config.batch_size:
        raise RuntimeError("Prefill replay fallito")
    return stats.transitions


def run_benchmarks(config: BenchmarkConfig) -> BenchmarkReport:
    """Esegue una prova per W; conserva stesso numero di ambienti e seed."""
    if not isinstance(config, BenchmarkConfig):
        raise TypeError("config deve essere BenchmarkConfig")

    # Gli import pesanti sono LOCALI: il main Windows-spawn non importa Torch
    # quando viene ricaricato nei worker attraverso __mp_main__.
    import torch
    from agent.dqn import DQNConfig, DQNTrainer
    from agent.network import QNetwork
    from agent.parallel_env import ParallelGridGames
    from agent.replay_buffer import ReplayBuffer
    from utils.seeds import (SeedPlan, derive_seed, make_torch_generator,
                             seed_everything)

    device = _resolve_device(config.device)
    if config.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 non supportato dalla GPU / build PyTorch")
    torch.set_num_threads(config.torch_threads)
    report_results: list[BenchmarkResult] = []
    print(f"[benchmark] device={device}, envs={config.envs}, "
          f"batch={config.batch_size}, workers={list(config.workers)}", flush=True)

    for workers in config.workers:
        seed_plan = SeedPlan(config.seed, config.experiment_id)
        seed_everything(seed_plan.experiment_seed, seed_cuda=device.type == "cuda")
        trainer = DQNTrainer(
            QNetwork(), device=device, config=DQNConfig(
                gamma=1.0, use_bfloat16_amp=config.bf16,
            ),
        )
        replay = ReplayBuffer(config.replay_capacity,
                              seed=derive_seed(seed_plan.experiment_seed, "dqn-benchmark-replay"))
        generator = make_torch_generator(seed_plan.policy_seed(), device=str(device))
        if device.type == "cuda":
            _sync(device)
            torch.cuda.reset_peak_memory_stats(device)

        start_startup = perf_counter()
        with ParallelGridGames(
            num_envs=config.envs, num_workers=workers, seed_plan=seed_plan,
            shaping_lambda=config.shaping_lambda,
        ) as environments:
            startup_seconds = max(perf_counter() - start_startup, 1e-12)
            if len(environments.worker_pids) != workers:
                raise RuntimeError("Numero di worker avviati errato")

            if config.warmup_simulation_steps:
                benchmark_simulation(environments, steps=config.warmup_simulation_steps)
            simulation = benchmark_simulation(environments, steps=config.simulation_steps)
            inference = benchmark_inference(
                environments.observation, trainer.online_network,
                device=device, iterations=config.inference_iters,
                warmup_iterations=config.warmup_inference_iters,
                epsilon=config.epsilon, generator=generator,
                pin_memory=config.pin_memory,
            )
            if config.warmup_collection_steps:
                from agent.collector import collect_parallel_experience
                collect_parallel_experience(
                    environments, trainer.online_network, replay,
                    steps_per_env=config.warmup_collection_steps,
                    epsilon=config.epsilon, device=device,
                    pin_memory=config.pin_memory, generator=generator,
                )
            collection = benchmark_collection(
                environments, trainer.online_network, replay,
                steps_per_env=config.steps_per_env, epsilon=config.epsilon,
                device=device, pin_memory=config.pin_memory, generator=generator,
            )
            prefill = _prefill_replay(environments, trainer, replay, config, generator)
            collection["extra_prefill_transitions_untimed"] = prefill
            updates = benchmark_updates(
                trainer, replay, updates=config.updates, batch_size=config.batch_size,
                warmup_updates=config.warmup_updates,
            )
            cycle = benchmark_cycle(
                environments, trainer, replay,
                steps_per_env=config.cycle_steps_per_env, updates=config.cycle_updates,
                batch_size=config.batch_size, epsilon=config.epsilon,
                pin_memory=config.pin_memory, generator=generator,
            )
            result = BenchmarkResult(
                workers=workers, envs=config.envs,
                worker_startup_seconds=startup_seconds,
                simulation=simulation, inference=inference,
                collection=collection, updates=updates, cycle=cycle,
                replay_bytes=replay.nbytes,
                gpu_peak_allocated_bytes=(torch.cuda.max_memory_allocated(device)
                                          if device.type == "cuda" else None),
                gpu_peak_reserved_bytes=(torch.cuda.max_memory_reserved(device)
                                         if device.type == "cuda" else None),
            )
            report_results.append(result)
        print(
            f"{workers:>2} worker | simulation {simulation['activations_per_second']:>9.1f} "
            f"att/s | inference {inference['observations_per_second']:>9.1f} obs/s | "
            f"collection {collection['transitions_per_second']:>9.1f} att/s | "
            f"updates {updates['updates_per_second']:>7.1f} upd/s | "
            f"cycle {cycle['activations_per_second']:>8.1f} att/s",
            flush=True,
        )
        report = BenchmarkReport(
            config=config, hardware=_hardware(device),
            results=tuple(report_results),
        )
        if config.output is not None:
            _write_report(report, config.output)

    return BenchmarkReport(config=config, hardware=_hardware(device),
                           results=tuple(report_results))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark reale Context-Conditioned Masked DQN")
    parser.add_argument("--device", default="cpu", help="cpu, cuda o cuda:0")
    parser.add_argument("--envs", "--num-envs", dest="envs", type=int, default=8)
    parser.add_argument("--workers", type=int, nargs="+", default=None,
                        help="Default: tra 8, 4, 2, 1, solo valori <= envs")
    parser.add_argument("--simulation-steps", type=int, default=100)
    parser.add_argument("--steps-per-env", "--collection-steps", dest="steps_per_env",
                        type=int, default=100)
    parser.add_argument("--inference-iters", type=int, default=100)
    parser.add_argument("--updates", type=int, default=32)
    parser.add_argument("--cycle-steps-per-env", type=int, default=None,
                        help="Default: uguale a --steps-per-env")
    parser.add_argument("--cycle-updates", type=int, default=32)
    parser.add_argument("--batch-size", "--minibatch-size", dest="batch_size",
                        type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=20_000)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--experiment-id", type=int, default=0)
    parser.add_argument("--shaping-lambda", type=float, default=0.1)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--warmup-simulation-steps", type=int, default=5)
    parser.add_argument("--warmup-inference-iters", type=int, default=10)
    parser.add_argument("--warmup-collection-steps", type=int, default=2)
    parser.add_argument("--warmup-updates", type=int, default=3)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", default="benchmarks/results/benchmark_dqn.json",
                        help="Percorso JSON; specificare '' per non salvare")
    args = parser.parse_args(argv)
    try:
        config = BenchmarkConfig(
            device=args.device, envs=args.envs,
            workers=tuple(args.workers if args.workers is not None
                          else [w for w in (8, 4, 2, 1) if w <= args.envs]),
            simulation_steps=args.simulation_steps,
            steps_per_env=args.steps_per_env,
            inference_iters=args.inference_iters, updates=args.updates,
            cycle_steps_per_env=(args.cycle_steps_per_env if args.cycle_steps_per_env is not None
                                 else args.steps_per_env),
            cycle_updates=args.cycle_updates, batch_size=args.batch_size,
            replay_capacity=args.replay_capacity, epsilon=args.epsilon,
            seed=args.seed, experiment_id=args.experiment_id,
            shaping_lambda=args.shaping_lambda, pin_memory=args.pin_memory,
            bf16=args.bf16, warmup_simulation_steps=args.warmup_simulation_steps,
            warmup_inference_iters=args.warmup_inference_iters,
            warmup_collection_steps=args.warmup_collection_steps,
            warmup_updates=args.warmup_updates, torch_threads=args.torch_threads,
            output=args.output or None,
        )
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    report = run_benchmarks(config)
    best = max(report.results, key=lambda result: result.cycle["activations_per_second"])
    print(f"[benchmark] ciclo piu' veloce in questa prova: {best.workers} worker "
          f"({best.cycle['activations_per_second']:.1f} att/s)")
    if config.output is not None:
        print(f"[benchmark] risultati: {Path(config.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["BenchmarkConfig", "BenchmarkResult", "BenchmarkReport",
           "benchmark_simulation", "benchmark_inference", "benchmark_collection",
           "benchmark_updates", "benchmark_cycle", "run_benchmarks", "main"]
