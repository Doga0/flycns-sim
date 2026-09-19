"""A three-cell excitatory chain; no downloaded graph required."""

import argparse
from pathlib import Path

import numpy as np

from malecns_sim.neural.backends.base import NeuralGraph
from malecns_sim.neural.backends.brian2_lif import Brian2LIFBackend
from malecns_sim.neural.config import load_config
from malecns_sim.neural.result import save_result
from malecns_sim.neural.stimulus import Stimulus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/neural/toy"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output exists: {args.output}; choose another --output")
    graph = NeuralGraph(
        np.array([101, 102, 103], dtype=np.uint64),
        np.arange(3, dtype=np.int32),
        np.array([0, 1], dtype=np.int32),
        np.array([1, 2], dtype=np.int32),
        np.array([250, 250], dtype=np.uint32),
        ["acetylcholine"] * 3,
        {"selection": {"type": "synthetic chain"}},
    )
    backend = Brian2LIFBackend(graph, load_config(), seed=args.seed, record_nodes=(0, 1, 2))
    backend.schedule(Stimulus((0,), start_ms=100, stop_ms=600))
    backend.run(1000)
    summary = save_result(backend, args.output, experiment="toy")
    print(f"Active neurons: {summary['active_neurons']}; spikes: {summary['total_spikes']}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
