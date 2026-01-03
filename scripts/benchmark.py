#!/usr/bin/env python3
"""Benchmarking script for performance analysis."""

import argparse
import time
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


def benchmark_operators(n_agents: int, batch_size: int, num_runs: int = 100):
    """Benchmark individual operators."""
    from src.operators.projection import AdaptiveProjection
    from src.operators.assignment import DifferentiableAssignment
    from src.operators.consensus import RobustConsensus

    rng = jax.random.PRNGKey(0)

    results = {}

    # Benchmark Projection
    proj = AdaptiveProjection()
    states = jax.random.normal(rng, (batch_size, n_agents, 64))
    params_proj = proj.init(rng, states)

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        output, _ = proj.apply(params_proj, states)
        output.block_until_ready()
        times.append((time.perf_counter() - start) * 1000)

    results['projection'] = {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
    }

    # Benchmark Assignment
    assign = DifferentiableAssignment()
    task_states = jax.random.normal(rng, (batch_size, n_agents, 64))
    params_assign = assign.init(rng, states, task_states)

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        output, _ = assign.apply(params_assign, states, task_states)
        output.block_until_ready()
        times.append((time.perf_counter() - start) * 1000)

    results['assignment'] = {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
    }

    # Benchmark Consensus
    consensus = RobustConsensus()
    params_consensus = consensus.init(rng, states)

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        output, _ = consensus.apply(params_consensus, states)
        output.block_until_ready()
        times.append((time.perf_counter() - start) * 1000)

    results['consensus'] = {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=str, default="ours")
    parser.add_argument("--agents", type=int, nargs="+", default=[100, 500, 1000])
    parser.add_argument("--output", type=str, default="results/benchmark_results.json")
    args = parser.parse_args()

    all_results = {}

    for n_agents in tqdm(args.agents, desc="Benchmarking"):
        results = benchmark_operators(n_agents, batch_size=32, num_runs=100)
        all_results[n_agents] = results

    # Save results
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
