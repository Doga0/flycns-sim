"""Inspect and visualize small neighborhoods in a built MaleCNS graph."""

import argparse
import heapq
from pathlib import Path

import numpy as np

from malecns_sim.cns.graph.store import CNSGraph


DEFAULT_GRAPH = Path(
    "data/processed/malecns-v1.0/published-v1"
)


def metadata_for_node(
    graph: CNSGraph,
    node_index: int,
) -> dict:
    return graph.nodes.slice(node_index, 1).to_pylist()[0]


def find_incoming(
    graph: CNSGraph,
    node_index: int,
    limit: int,
    chunk_size: int = 1_000_000,
) -> list[tuple[int, int]]:
    """
    Find the strongest incoming edges.

    v0.2 stores outgoing CSR only, so incoming edges are found
    by scanning dst.npy in bounded chunks.

    Only the strongest `limit` edges are kept in memory.
    """

    if limit <= 0:
        return []

    src_array = graph.arrays["src"]
    dst_array = graph.arrays["dst"]
    contacts_array = graph.arrays["contacts"]

    heap: list[tuple[int, int]] = []

    for start in range(0, len(dst_array), chunk_size):
        end = min(start + chunk_size, len(dst_array))

        dst_chunk = dst_array[start:end]

        matches = np.flatnonzero(
            dst_chunk == node_index
        )

        for local_position in matches:
            position = start + int(local_position)

            source = int(src_array[position])
            weight = int(contacts_array[position])

            item = (weight, source)

            if len(heap) < limit:
                heapq.heappush(heap, item)

            elif item > heap[0]:
                heapq.heapreplace(heap, item)

    # strongest first
    return [
        (source, weight)
        for weight, source in sorted(
            heap,
            reverse=True,
        )
    ]


def find_outgoing(
    graph: CNSGraph,
    node_index: int,
    limit: int,
) -> list[tuple[int, int]]:
    """Return strongest outgoing connections."""

    if limit <= 0:
        return []

    targets, contacts = graph.outgoing(node_index)

    if len(targets) == 0:
        return []

    count = min(limit, len(targets))

    if len(targets) <= count:
        selected = np.arange(len(targets))

    else:
        selected = np.argpartition(
            contacts,
            -count,
        )[-count:]

    selected = selected[
        np.argsort(
            contacts[selected],
            kind="stable",
        )[::-1]
    ]

    return [
        (
            int(targets[position]),
            int(contacts[position]),
        )
        for position in selected
    ]


def node_label(
    graph: CNSGraph,
    node_index: int,
    include_type: bool = False,
) -> str:
    label = f"N{node_index}"

    if not include_type:
        return label

    metadata = metadata_for_node(
        graph,
        node_index,
    )

    neuron_type = metadata.get("type")

    if neuron_type is None:
        return label

    neuron_type = str(neuron_type).strip()

    if not neuron_type:
        return label

    if len(neuron_type) > 24:
        neuron_type = neuron_type[:21] + "..."

    return f"{label}\n{neuron_type}"


def get_neighborhood(
    graph: CNSGraph,
    node_index: int,
    incoming_limit: int,
    outgoing_limit: int,
) -> dict:
    """Collect the small 1-hop neighborhood used by text and plots."""

    incoming = find_incoming(
        graph,
        node_index,
        incoming_limit,
    )

    outgoing = find_outgoing(
        graph,
        node_index,
        outgoing_limit,
    )

    all_targets, all_contacts = graph.outgoing(
        node_index
    )

    return {
        "node_index": node_index,
        "body_id": int(
            graph.body_ids[node_index]
        ),
        "incoming": incoming,
        "outgoing": outgoing,
        "out_degree": len(all_targets),
        "outgoing_contacts": int(
            all_contacts.sum(
                dtype=np.uint64
            )
        ),
    }


def print_neighborhood(
    graph: CNSGraph,
    neighborhood: dict,
) -> None:
    node_index = neighborhood["node_index"]

    print()
    print("=" * 72)
    print("MaleCNS 1-hop neighborhood")
    print("=" * 72)

    print(
        f"node_index       : "
        f"{node_index}"
    )
    print(
        f"body_id          : "
        f"{neighborhood['body_id']}"
    )
    print(
        f"total out_degree : "
        f"{neighborhood['out_degree']:,}"
    )
    print(
        f"outgoing contacts: "
        f"{neighborhood['outgoing_contacts']:,}"
    )

    print()
    print("Incoming:")
    print()

    incoming = neighborhood["incoming"]

    if not incoming:
        print("    No incoming edges shown.")
    else:
        for source, weight in incoming:
            print(
                f"    N{source}"
                f" ──{weight}──▶ "
                f"[ N{node_index} ]"
            )

    print()
    print(f"[ N{node_index} ]")

    outgoing = neighborhood["outgoing"]

    if not outgoing:
        print("    └── no outgoing edges")
        return

    for number, (target, weight) in enumerate(
        outgoing
    ):
        last = number == len(outgoing) - 1

        branch = "└" if last else "├"

        print(
            f"    {branch}──{weight}──▶ "
            f"N{target}"
        )


def print_metadata(
    graph: CNSGraph,
    node_index: int,
) -> None:
    metadata = metadata_for_node(
        graph,
        node_index,
    )

    print()
    print("=" * 72)
    print("Metadata")
    print("=" * 72)

    for key, value in metadata.items():
        print(f"{key}: {value}")


def edge_width(
    weight: int,
    maximum: int,
) -> float:
    """
    Compress large synapse-count differences so very strong edges
    do not make weaker edges visually disappear.
    """

    if maximum <= 1:
        return 1.5

    normalized = (
        np.log1p(weight)
        / np.log1p(maximum)
    )

    return 0.8 + 2.5 * float(normalized)


def vertical_positions(
    count: int,
) -> np.ndarray:
    if count == 0:
        return np.empty(0)

    if count == 1:
        return np.array([0.0])

    span = max(
        2.0,
        count * 0.45,
    )

    return np.linspace(
        span / 2,
        -span / 2,
        count,
    )


def plot_neighborhood(
    graph: CNSGraph,
    neighborhood: dict,
    *,
    save_path: Path | None,
    show: bool,
) -> None:
    """
    Draw only the selected 1-hop subgraph.

    The full MaleCNS graph is never copied into NetworkX or a dense matrix.
    """

    import matplotlib.pyplot as plt

    center = neighborhood["node_index"]
    body_id = neighborhood["body_id"]

    incoming = neighborhood["incoming"]
    outgoing = neighborhood["outgoing"]

    all_weights = (
        [weight for _, weight in incoming]
        + [weight for _, weight in outgoing]
    )

    max_weight = max(
        all_weights,
        default=1,
    )

    incoming_y = vertical_positions(
        len(incoming)
    )

    outgoing_y = vertical_positions(
        len(outgoing)
    )

    center_x = 0.0
    center_y = 0.0

    incoming_x = -2.5
    outgoing_x = 2.5

    height = max(
        6.0,
        0.65 * max(
            len(incoming),
            len(outgoing),
            5,
        ),
    )

    fig, ax = plt.subplots(
        figsize=(13, height)
    )

    #
    # Center node
    #
    ax.scatter(
        [center_x],
        [center_y],
        s=1400,
        zorder=5,
    )

    ax.text(
        center_x,
        center_y,
        node_label(
            graph,
            center,
            include_type=True,
        ),
        ha="center",
        va="center",
        zorder=6,
    )

    #
    # Incoming nodes
    #
    for (
        (source, weight),
        y,
    ) in zip(
        incoming,
        incoming_y,
    ):
        ax.scatter(
            [incoming_x],
            [y],
            s=750,
            zorder=5,
        )

        ax.text(
            incoming_x,
            y,
            node_label(
                graph,
                source,
                include_type=True,
            ),
            ha="center",
            va="center",
            zorder=6,
        )

        width = edge_width(
            weight,
            max_weight,
        )

        ax.annotate(
            "",
            xy=(center_x - 0.18, center_y),
            xytext=(incoming_x + 0.18, y),
            arrowprops={
                "arrowstyle": "->",
                "linewidth": width,
                "alpha": 0.75,
            },
            zorder=2,
        )

        middle_x = (
            incoming_x + center_x
        ) / 2

        middle_y = (
            y + center_y
        ) / 2

        ax.text(
            middle_x,
            middle_y + 0.08,
            str(weight),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    #
    # Outgoing nodes
    #
    for (
        (target, weight),
        y,
    ) in zip(
        outgoing,
        outgoing_y,
    ):
        ax.scatter(
            [outgoing_x],
            [y],
            s=750,
            zorder=5,
        )

        ax.text(
            outgoing_x,
            y,
            node_label(
                graph,
                target,
                include_type=True,
            ),
            ha="center",
            va="center",
            zorder=6,
        )

        width = edge_width(
            weight,
            max_weight,
        )

        ax.annotate(
            "",
            xy=(outgoing_x - 0.18, y),
            xytext=(center_x + 0.18, center_y),
            arrowprops={
                "arrowstyle": "->",
                "linewidth": width,
                "alpha": 0.75,
            },
            zorder=2,
        )

        middle_x = (
            outgoing_x + center_x
        ) / 2

        middle_y = (
            y + center_y
        ) / 2

        ax.text(
            middle_x,
            middle_y + 0.08,
            str(weight),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.text(
        incoming_x,
        max(
            incoming_y,
            default=0,
        )
        + 0.8,
        "Incoming",
        ha="center",
        fontsize=12,
    )

    ax.text(
        outgoing_x,
        max(
            outgoing_y,
            default=0,
        )
        + 0.8,
        "Outgoing",
        ha="center",
        fontsize=12,
    )

    ax.set_title(
        f"MaleCNS 1-hop neighborhood\n"
        f"N{center} · body_id={body_id} · "
        f"total out-degree={neighborhood['out_degree']:,}"
    )

    max_span = max(
        len(incoming),
        len(outgoing),
        5,
    )

    y_limit = max(
        2.0,
        max_span * 0.35,
    )

    ax.set_xlim(
        -3.6,
        3.6,
    )

    ax.set_ylim(
        -y_limit - 0.8,
        y_limit + 1.2,
    )

    ax.axis("off")

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=180,
            bbox_inches="tight",
        )

        print(
            f"Plot saved: {save_path}"
        )

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect and visualize MaleCNS "
            "local connectivity."
        )
    )

    parser.add_argument(
        "--graph",
        type=Path,
        default=DEFAULT_GRAPH,
        help=(
            "Built graph directory. "
            f"Default: {DEFAULT_GRAPH}"
        ),
    )

    selector = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    selector.add_argument(
        "--node",
        type=int,
        help="Internal contiguous node index.",
    )

    selector.add_argument(
        "--body-id",
        type=int,
        help="Original MaleCNS body ID.",
    )

    parser.add_argument(
        "--incoming",
        type=int,
        default=5,
        help=(
            "Number of strongest incoming "
            "edges to show. Default: 5"
        ),
    )

    parser.add_argument(
        "--outgoing",
        type=int,
        default=8,
        help=(
            "Number of strongest outgoing "
            "edges to show. Default: 8"
        ),
    )

    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Print all metadata for the center node.",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Open the neighborhood plot in a window.",
    )

    parser.add_argument(
        "--save-plot",
        type=Path,
        help="Save the neighborhood plot to PNG/PDF/SVG.",
    )

    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Do not print the ASCII neighborhood.",
    )

    args = parser.parse_args()

    if args.incoming < 0:
        parser.error(
            "--incoming must be >= 0"
        )

    if args.outgoing < 0:
        parser.error(
            "--outgoing must be >= 0"
        )

    if not args.graph.is_dir():
        parser.error(
            f"Graph directory does not exist: "
            f"{args.graph}"
        )

    with CNSGraph.load(
        args.graph
    ) as graph:

        if args.node is not None:
            node_index = args.node

        else:
            try:
                node_index = graph.node_index(
                    args.body_id
                )

            except KeyError:
                parser.error(
                    f"body_id {args.body_id} "
                    "is not present in this graph."
                )

        if not (
            0
            <= node_index
            < len(graph.body_ids)
        ):
            parser.error(
                f"node must be between 0 and "
                f"{len(graph.body_ids) - 1}"
            )

        print(
            "Querying neighborhood...",
            flush=True,
        )

        neighborhood = get_neighborhood(
            graph,
            node_index,
            incoming_limit=args.incoming,
            outgoing_limit=args.outgoing,
        )

        if not args.no_text:
            print_neighborhood(
                graph,
                neighborhood,
            )

        if args.metadata:
            print_metadata(
                graph,
                node_index,
            )

        if args.plot or args.save_plot:
            plot_neighborhood(
                graph,
                neighborhood,
                save_path=args.save_plot,
                show=args.plot,
            )


if __name__ == "__main__":
    main()