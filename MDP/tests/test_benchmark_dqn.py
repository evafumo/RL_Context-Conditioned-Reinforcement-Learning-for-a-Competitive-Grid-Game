"""Verifiche CPU del benchmark; non richiede CUDA."""
from dataclasses import asdict
import json
from pathlib import Path
import pytest

from benchmarks.benchmark_dqn import (
    BenchmarkConfig, _expected_terminals, _write_report,
    benchmark_collection, benchmark_cycle, benchmark_inference,
    benchmark_simulation, benchmark_updates, run_benchmarks,
)


def test_config_guards():
    assert BenchmarkConfig(envs=8).workers == (8, 4, 2, 1)
    with pytest.raises(ValueError):
        BenchmarkConfig(envs=2, workers=(3,))
    with pytest.raises(ValueError):
        BenchmarkConfig(envs=2, workers=(1, 1))
    with pytest.raises(ValueError):
        BenchmarkConfig(envs=2, workers=(1,), batch_size=11, replay_capacity=10)
    with pytest.raises(ValueError):
        BenchmarkConfig(envs=2, workers=(1,), device='cpu', pin_memory=True)
    with pytest.raises(ValueError):
        BenchmarkConfig(envs=2, workers=(1,), device='cpu', bf16=True)
    with pytest.raises(ValueError):
        BenchmarkConfig(envs=2, workers=(1,), epsilon=1.1)
    with pytest.raises(ValueError):
        BenchmarkConfig(envs=2, workers=(1,), simulation_steps=0)


def test_terminal_count_formula():
    assert _expected_terminals([1, 100, 10, 80], 1) == 1
    assert _expected_terminals([1, 100, 10, 80], 100) == 4
    assert _expected_terminals([99, 100], 101) == 3


def test_full_benchmark_reproducible_shapes(tmp_path):
    import torch
    torch.set_num_threads(1)
    config = BenchmarkConfig(
        envs=2, workers=(1, 2), simulation_steps=100,
        steps_per_env=100, cycle_steps_per_env=100,
        inference_iters=2, updates=2, cycle_updates=2,
        batch_size=16, replay_capacity=512, warmup_simulation_steps=0,
        warmup_inference_iters=1, warmup_collection_steps=0,
        warmup_updates=1, output=tmp_path/'results.json',
    )
    report = run_benchmarks(config)
    saved = json.loads((tmp_path/'results.json').read_text())
    assert saved['config']['device'] == 'cpu'
    assert len(report.results) == len(saved['results']) == 2
    for result in report.results:
        assert result.simulation['activations'] == 200
        assert result.simulation['completed_episodes'] == 2
        assert result.collection['transitions'] == 200
        assert result.collection['completed_episodes'] == 2
        assert result.collection['agent_decisions'] + result.collection['forced_activations'] == 200
        assert result.cycle['transitions'] == 200
        assert result.cycle['completed_episodes'] == 2
        assert result.cycle['agent_decisions'] + result.cycle['forced_activations'] == 200
        assert result.updates['updates'] == 2
        assert result.inference['observations'] == 4
        assert result.gpu_peak_allocated_bytes is None
        assert result.gpu_peak_reserved_bytes is None
        assert result.replay_bytes > 0
        assert result.worker_startup_seconds > 0
        assert result.cycle['activations_per_second'] > 0


def test_prefill_and_path_output(tmp_path):
    config = BenchmarkConfig(
        envs=1, workers=(1,), simulation_steps=1, steps_per_env=1,
        cycle_steps_per_env=1, inference_iters=1, updates=1,
        cycle_updates=1, batch_size=8, replay_capacity=12,
        warmup_simulation_steps=0, warmup_inference_iters=0,
        warmup_collection_steps=0, warmup_updates=0,
        output=tmp_path/'out.json',
    )
    report = run_benchmarks(config)
    assert report.results[0].collection['extra_prefill_transitions_untimed'] == 7
    assert json.loads((tmp_path/'out.json').read_text())['config']['output'] == str(config.output)
