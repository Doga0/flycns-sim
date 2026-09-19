"""Stimulate an anatomical sensory population and classify full-CNS spike activity."""

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from malecns_sim.cns.graph.store import CNSGraph
from malecns_sim.cns.graph.validate import validate_graph
from malecns_sim.io.catalogue import IOCatalog
from malecns_sim.io.inspect import DEFAULT_CATALOGUE
from malecns_sim.io.roles import NeuronRole
from malecns_sim.io.validate import validate_catalogue
from malecns_sim.neural.backends.base import NeuralGraph
from malecns_sim.neural.backends.brian2_lif import Brian2LIFBackend
from malecns_sim.neural.config import load_config
from malecns_sim.neural.experiments.common import DEFAULT_GRAPH
from malecns_sim.neural.result import save_result, validate_result
from malecns_sim.neural.stimulus import Stimulus


def _activity_tables(catalogue: IOCatalog, backend, stimulated: np.ndarray, duration_ms: float):
    _, spike_nodes = backend.get_spikes()
    counts = np.bincount(spike_nodes, minlength=catalogue.neurons.num_rows).astype(np.uint64)
    frame = catalogue.neurons.to_pandas()
    frame["spike_count"] = counts
    frame["rate_hz"] = counts / (duration_ms / 1000)
    stimulated_mask = np.zeros(len(frame), dtype=bool)
    stimulated_mask[stimulated] = True
    frame["is_stimulated"] = stimulated_mask
    active = frame.loc[
        counts > 0,
        [
            "node_index",
            "body_id",
            "role",
            "role_source_value",
            "superclass",
            "class",
            "subclass",
            "type",
            "side",
            "nerve",
            "body_region",
            "sensory_system",
            "spike_count",
            "rate_hz",
            "is_stimulated",
        ],
    ]
    role_rows = []
    for role in NeuronRole:
        mask = frame["role"].eq(role.value).to_numpy()
        role_rows.append(
            {
                "role": role.value,
                "catalogue_neurons": int(mask.sum()),
                "stimulated_neurons": int(np.count_nonzero(mask & stimulated_mask)),
                "active_neurons": int(np.count_nonzero(mask & (counts > 0))),
                "total_spikes": int(counts[mask].sum()),
            }
        )
    return (
        pa.Table.from_pandas(active, preserve_index=False).replace_schema_metadata(None),
        pa.Table.from_pylist(role_rows),
        role_rows,
    )


def validate_io_experiment(output: str | Path, catalogue: IOCatalog) -> dict:
    summary = validate_result(output)
    output = Path(output)
    active = pq.read_table(output / "active_neurons.parquet")
    role_activity = pq.read_table(output / "role_activity.parquet")
    catalogue_lookup = {
        int(row["node_index"]): row
        for row in catalogue.neurons.select(
            ["node_index", "body_id", "role", "type", "side"]
        ).to_pylist()
    }
    for row in active.select(
        ["node_index", "body_id", "role", "type", "side", "spike_count"]
    ).to_pylist():
        source = catalogue_lookup[row["node_index"]]
        if any(row[field] != source[field] for field in ("body_id", "role", "type", "side")):
            raise ValueError("Active-neuron metadata differs from the I/O catalogue")
        if row["spike_count"] <= 0:
            raise ValueError("Active-neuron table contains a zero-spike neuron")
    rows = role_activity.to_pylist()
    if {row["role"] for row in rows} != {role.value for role in NeuronRole}:
        raise ValueError("Role activity does not cover every normalized role")
    if sum(row["active_neurons"] for row in rows) != summary["active_neurons"]:
        raise ValueError("Role activity does not preserve active-neuron count")
    if sum(row["total_spikes"] for row in rows) != summary["total_spikes"]:
        raise ValueError("Role activity does not preserve total spike count")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--class", dest="cell_class", default="mechanosensory_proprioceptive")
    parser.add_argument("--subclass", default="leg")
    parser.add_argument("--side")
    parser.add_argument("--duration-ms", type=float, default=100)
    parser.add_argument("--rate-hz", type=float, default=100)
    parser.add_argument("--gain-mv", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/io/leg_proprio_seed42_100ms"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists; choose another --output: {args.output}")
    config = load_config(args.config)
    if config.steps(args.duration_ms) <= 0:
        parser.error("--duration-ms must be positive")
    criteria = {"class": args.cell_class, "subclass": args.subclass}
    if args.side:
        criteria["side"] = args.side
    try:
        validate_catalogue(args.catalogue, args.graph)
        catalogue = IOCatalog.load(args.catalogue)
        selected = catalogue.sensory(**criteria)
        if not len(selected):
            raise ValueError(f"No sensory neurons match {criteria}")
        with CNSGraph.load(args.graph) as stored:
            validate_graph(stored)
            if not np.array_equal(selected.body_ids, stored.body_ids[selected.node_indices]):
                raise ValueError("Sensory selection does not map exactly to the graph")
            graph = NeuralGraph.from_graph(stored)
            backend = Brian2LIFBackend(graph, config, seed=args.seed)
            gain = args.gain_mv if args.gain_mv is not None else config.stimulus["gain_mv"]
            backend.schedule(
                Stimulus(
                    tuple(int(node) for node in selected.node_indices),
                    rate_hz=args.rate_hz,
                    start_ms=0,
                    stop_ms=args.duration_ms,
                    gain_mv=gain,
                )
            )
            print(
                f"Selected sensory neurons: {len(selected):,}\n"
                f"Selection: class={args.cell_class!r}, subclass={args.subclass!r}, "
                f"side={args.side!r}\nRunning full MaleCNS LIF for {args.duration_ms:g} ms...",
                flush=True,
            )
            backend.run(args.duration_ms)
            active, role_table, role_rows = _activity_tables(
                catalogue, backend, selected.node_indices, args.duration_ms
            )
            extra = {
                "catalogue_manifest": catalogue.manifest,
                "sensory_selection": {"role": "sensory", **criteria},
                "interpretation": (
                    "Anatomically classified connectivity-driven activity; no FlyGym sensor "
                    "encoding, motor decoding, or behavioral validation."
                ),
                "role_activity": role_rows,
            }
            summary = save_result(
                backend,
                args.output,
                experiment="io-catalogue-sensory-propagation",
                extra=extra,
                extra_tables={
                    "active_neurons.parquet": active,
                    "role_activity.parquet": role_table,
                },
            )
            validate_io_experiment(args.output, catalogue)
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"I/O experiment failed: {error}") from None
    print("Simulation completed")
    for row in role_rows:
        print(f"{row['role']:<12} active={row['active_neurons']:<6} spikes={row['total_spikes']}")
    print(f"Total active neurons: {summary['active_neurons']}")
    print(f"Total spikes: {summary['total_spikes']}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
