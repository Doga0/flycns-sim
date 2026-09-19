"""CLI orchestration shared by the full graph and bounded subgraph experiments."""

import argparse
from pathlib import Path

import numpy as np

from malecns_sim.cns.graph.store import CNSGraph
from malecns_sim.cns.graph.validate import validate_graph
from malecns_sim.neural.backends.base import NeuralGraph
from malecns_sim.neural.backends.brian2_lif import Brian2LIFBackend
from malecns_sim.neural.config import load_config
from malecns_sim.neural.experiments.selection import choose_source, extract_subgraph
from malecns_sim.neural.result import save_result
from malecns_sim.neural.stimulus import Stimulus

DEFAULT_GRAPH = Path("data/processed/malecns-v1.0/published-v1")


def main(mode: str, argv=None, *, spontaneous: bool = False) -> None:
    try:
        _run(mode, argv, spontaneous=spontaneous)
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"Neural experiment failed: {error}") from None


def _run(mode: str, argv=None, *, spontaneous: bool = False) -> None:
    parser = argparse.ArgumentParser(description=f"Shiu-derived LIF {mode} experiment")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--duration-ms", type=float, default=100 if mode == "full_cns" else 1000)
    parser.add_argument("--stimulate-body-id", type=int)
    parser.add_argument("--rate-hz", type=float)
    parser.add_argument("--gain-mv", type=float)
    parser.add_argument("--start-ms", type=float)
    parser.add_argument("--stop-ms", type=float)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-stimulus", action="store_true", default=spontaneous)
    if mode == "subgraph":
        parser.add_argument("--max-nodes", type=int, default=5000)
        parser.add_argument("--hops", type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config.steps(args.duration_ms) <= 0:
        parser.error("--duration-ms must be positive")
    start = args.start_ms if args.start_ms is not None else (0 if mode == "full_cns" else 100)
    stop = (
        args.stop_ms
        if args.stop_ms is not None
        else (args.duration_ms if mode == "full_cns" else 600)
    )
    if not args.no_stimulus and not 0 <= start < stop <= args.duration_ms:
        parser.error("Stimulus window must fit inside duration; set --start-ms and --stop-ms")
    output = (
        args.output or Path("outputs/neural") / f"{mode}_seed{args.seed}_{args.duration_ms:g}ms"
    )
    if output.exists():
        parser.error(f"Output already exists; choose another --output: {output}")
    print("Validating MaleCNS graph and artifact checksums...", flush=True)
    with CNSGraph.load(args.graph) as stored:
        validate_graph(stored)
        source = None
        if mode == "subgraph" or not args.no_stimulus:
            source = (
                stored.node_index(args.stimulate_body_id)
                if args.stimulate_body_id is not None
                else choose_source(stored, config)
            )
        graph = (
            NeuralGraph.from_graph(stored)
            if mode == "full_cns"
            else extract_subgraph(stored, source, max_nodes=args.max_nodes, hops=args.hops)
        )
        print(
            f"MaleCNS graph loaded\nNeurons: {len(graph.body_ids):,}\n"
            f"Connections: {len(graph.src):,}\nModel: Shiu-derived LIF\n"
            f"dt: {config.dt_ms} ms\nduration: {args.duration_ms} ms",
            flush=True,
        )
        stimulus = None
        if not args.no_stimulus:
            local_source = int(np.flatnonzero(graph.node_indices == source)[0])
            stimulus = Stimulus(
                (local_source,),
                rate_hz=args.rate_hz if args.rate_hz is not None else config.stimulus["rate_hz"],
                gain_mv=args.gain_mv if args.gain_mv is not None else config.stimulus["gain_mv"],
                start_ms=start,
                stop_ms=stop,
            )
            stimulus.validate(len(graph.body_ids), config)
            print(f"Stimulated body ID: {int(graph.body_ids[local_source])}", flush=True)
        print("Constructing Brian2 neurons and synapses...", flush=True)
        backend = Brian2LIFBackend(graph, config, seed=args.seed)
        if stimulus:
            backend.schedule(stimulus)
        print(
            f"Stimulated neurons: {0 if stimulus is None else 1}\nRunning simulation...", flush=True
        )
        backend.run(args.duration_ms)
        extra = {
            "source_selection": "explicit body ID"
            if args.stimulate_body_id is not None
            else "strongest positive-to-positive non-self edge, canonical COO tie order"
        }
        if source is None:
            extra["source_selection"] = "none (no external stimulus)"
        if mode == "subgraph":
            hops = np.array(graph.provenance["selection"]["local_node_hops"])
            _, spike_nodes = backend.get_spikes()
            extra["hop_activity"] = [
                {
                    "hop": hop,
                    "active_neurons": int(np.count_nonzero(hops[np.unique(spike_nodes)] == hop)),
                    "spikes": int(np.count_nonzero(hops[spike_nodes] == hop)),
                }
                for hop in range(args.hops + 1)
            ]
        summary = save_result(backend, output, experiment=mode, extra=extra)
        print(
            f"Simulation completed\nActive neurons: {summary['active_neurons']}\n"
            f"Active unstimulated neurons: {summary['active_unstimulated_neurons']}\n"
            f"Total spikes: {summary['total_spikes']}\nResults: {output}",
            flush=True,
        )
