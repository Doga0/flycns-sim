"""Save spike raster, firing rates and optional membrane traces from a neural result."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from malecns_sim.neural.result import validate_result


def next_spike_latencies(source, target) -> tuple[np.ndarray, np.ndarray]:
    """Pair each source with the strictly next target; targets may be reused.

    This descriptive timing statistic does not establish causal transmission.
    Source spikes without a subsequent target are excluded.
    """
    source, target = np.sort(source), np.sort(target)
    indices = np.searchsorted(target, source, side="right")
    matched = indices < len(target)
    return source[matched], target[indices[matched]] - source[matched]


def visualize(
    directory: Path,
    *,
    output: Path | None = None,
    max_nodes: int = 32,
    toy_latency: bool = False,
) -> list[Path]:
    """Plot at most max_nodes cells, prioritizing driven cells then highest spike counts."""
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    directory = Path(directory)
    summary = validate_result(directory)
    experiment = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    mapping = pq.read_table(directory / "node_mapping.parquet").to_pandas()
    rates = pq.read_table(directory / "rates.parquet").to_pandas()
    if toy_latency and (experiment["experiment"] != "toy" or len(mapping) != 3 or max_nodes < 3):
        raise ValueError("--toy-latency requires a three-neuron toy result and --max-nodes >= 3")
    driven = {node for s in experiment["stimuli"] for node in s["nodes"]}
    selected = sorted(
        range(len(mapping)),
        key=lambda i: (i not in driven, -int(rates["spike_count"].iloc[i]), i),
    )[:max_nodes]
    selected.sort()
    bodies = [int(mapping["body_id"].iloc[i]) for i in selected]
    spikes = pq.read_table(
        directory / "spikes.parquet",
        filters=[("node_index", "in", mapping["node_index"].iloc[selected].tolist())],
    ).to_pandas()
    labels = [
        f"Neuron {i} / body {body}" + (" — stimulated" if i in driven else "")
        for i, body in zip(selected, bodies)
    ]
    colors = [f"C{i % 10}" for i in range(len(selected))]
    output = Path(output) if output is not None else directory / "plots"
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    subset = f" ({len(selected)}/{len(mapping)} neurons)"

    def figure(height):
        fig = Figure(figsize=(12, height), layout="constrained")
        FigureCanvasAgg(fig)
        return fig, fig.subplots()

    def time_axis(ax):
        for number, stimulus in enumerate(experiment["stimuli"]):
            stop = stimulus["stop_ms"]
            ax.axvspan(
                stimulus["start_ms"],
                experiment["duration_ms"] if stop is None else stop,
                color="gray",
                alpha=0.15,
                label="External stimulus window" if number == 0 else None,
            )
        ax.set(xlim=(0, experiment["duration_ms"]), xlabel="Time (ms)")

    def save(fig, name):
        path = output / name
        fig.savefig(path, dpi=160)
        fig.clear()
        paths.append(path)

    fig, ax = figure(max(4, len(selected) * 0.3 + 2))
    events = [np.sort(spikes.loc[spikes["body_id"] == body, "t_ms"]) for body in bodies]
    ax.eventplot(events, colors=colors, linelengths=0.7)
    ax.set(
        yticks=range(len(selected)),
        yticklabels=labels,
        ylabel="Neuron",
        title="Spike raster" + subset,
        ylim=(-0.7, len(selected) - 0.3),
    )
    time_axis(ax)
    if experiment["stimuli"]:
        ax.legend(loc="upper right")
    save(fig, "spike_raster.png")

    fig, ax = figure(max(4, len(selected) * 0.3 + 2))
    values = rates["rate_hz"].iloc[selected].to_numpy()
    bars = ax.barh(labels, values, color=colors)
    ax.bar_label(
        bars,
        labels=[
            f"{rate:.2f} Hz / {int(rates['spike_count'].iloc[i])} spikes"
            for i, rate in zip(selected, values)
        ],
        padding=4,
    )
    ax.set(
        xlabel="Firing rate (Hz; full simulation duration)",
        title="Firing rates" + subset,
        xlim=(0, max(float(values.max()), 1) * 1.45),
    )
    save(fig, "firing_rates.png")

    if "state_trace.parquet" in summary["artifacts"]:
        trace = pq.read_table(
            directory / "state_trace.parquet", filters=[("local_index", "in", selected)]
        ).to_pandas()
        if not trace.empty:
            fig, ax = figure(6)
            for i, label, color in zip(selected, labels, colors):
                rows = trace.loc[trace["local_index"] == i].sort_values("t_ms")
                if not rows.empty:
                    ax.plot(rows["t_ms"], rows["v_mv"], label=label, color=color, lw=0.8)
            ax.axhline(
                experiment["config"]["v_threshold_mv"],
                color="black",
                ls="--",
                label="Spike threshold",
            )
            time_axis(ax)
            ax.set(
                ylabel="Membrane potential (mV)",
                title="LIF membrane dynamics (end-of-step samples, after reset)",
            )
            ax.legend(fontsize="small")
            save(fig, "membrane_potential.png")
    if toy_latency:
        fig, ax = figure(5)
        rows = []
        for source, target in ((0, 1), (1, 2), (0, 2)):
            times, latencies = next_spike_latencies(events[source], events[target])
            label = f"N{source} → N{target}"
            if len(latencies):
                ax.hist(latencies, bins=20, histtype="step", label=f"{label} (n={len(times)})")
            rows.extend((source, target, t, delay) for t, delay in zip(times, latencies))
        ax.set(
            xlabel="Time to strictly next target spike (ms)",
            ylabel="Pair count",
            title="Toy next-spike latency (descriptive pairing; not a causal estimate)",
        )
        if rows:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No subsequent target spikes", transform=ax.transAxes, ha="center")
        save(fig, "next_spike_latency.png")
        path = output / "next_spike_latency.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["source_local_index", "target_local_index", "source_t_ms", "latency_ms"]
            )
            writer.writerows(rows)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, help="Plot directory (default: RESULT/plots)")
    parser.add_argument("--max-nodes", type=int, default=32)
    parser.add_argument(
        "--toy-latency",
        action="store_true",
        help="Also save descriptive next-spike latency histogram and CSV",
    )
    args = parser.parse_args()
    try:
        paths = visualize(
            args.directory,
            output=args.output,
            max_nodes=args.max_nodes,
            toy_latency=args.toy_latency,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Visualization failed: {error}\n")
    for path in paths:
        print(path)
    if not any(path.name == "membrane_potential.png" for path in paths):
        print("No state trace for selected neurons; rerun with record_nodes to plot voltage.")


if __name__ == "__main__":
    main()
