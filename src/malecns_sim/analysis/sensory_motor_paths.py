"""Anatomical sensory-to-motor paths under an explicit LIF sign policy; no dynamics.

A path of ``hops`` synapses visits ``hops + 1`` neurons. Multiplicity is counted over
walks (a neuron may repeat); short paths are also listed one by one, flagged when a
neuron repeats. The sign of a path depends only on its presynaptic neurons (the source
and every intermediate), exactly as `Brian2LIFBackend` signs each synapse by its source.
"""

import math
from dataclasses import dataclass

import numpy as np

UNREACHED = -1
# Order matters: index into the propagated count vectors.
SIGN_CLASSES = ("excitatory", "net_excitatory_via_inhibition", "net_inhibitory", "muted")


@dataclass(frozen=True)
class Connectivity:
    """COO plus outgoing CSR views; `CNSGraph` arrays are used without copying."""

    num_nodes: int
    src: np.ndarray
    dst: np.ndarray
    contacts: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    csr_contacts: np.ndarray

    @classmethod
    def from_graph(cls, graph):
        a = graph.arrays
        return cls(
            len(graph.body_ids),
            a["src"],
            a["dst"],
            a["contacts"],
            a["csr_indptr"],
            a["csr_indices"],
            a["csr_contacts"],
        )

    @classmethod
    def from_edges(cls, num_nodes, edges):
        """Small synthetic graphs: ``edges`` is an iterable of (src, dst, contacts)."""
        edges = sorted(edges)
        src = np.array([e[0] for e in edges], dtype=np.int32)
        dst = np.array([e[1] for e in edges], dtype=np.int32)
        contacts = np.array([e[2] for e in edges], dtype=np.uint32)
        indptr = np.zeros(num_nodes + 1, dtype=np.int64)
        np.add.at(indptr, src + 1, 1)
        return cls(num_nodes, src, dst, contacts, np.cumsum(indptr), dst.copy(), contacts.copy())


def _validate_nodes(nodes, num_nodes, name):
    nodes = np.asarray(nodes, dtype=np.int64)
    if nodes.ndim != 1 or not len(nodes) or len(np.unique(nodes)) != len(nodes):
        raise ValueError(f"{name} must be a nonempty list of unique node indices")
    if np.any(nodes < 0) or np.any(nodes >= num_nodes):
        raise ValueError(f"{name} contains an unknown node index")
    return nodes


def hop_distances(conn: Connectivity, starts, max_hops: int, *, reverse=False) -> np.ndarray:
    """Fewest synapses from `starts` (or to them, if reverse); UNREACHED beyond max_hops."""
    starts = _validate_nodes(starts, conn.num_nodes, "starts")
    a, b = (conn.dst, conn.src) if reverse else (conn.src, conn.dst)
    distance = np.full(conn.num_nodes, UNREACHED, dtype=np.int16)
    distance[starts] = 0
    frontier = np.zeros(conn.num_nodes, dtype=bool)
    frontier[starts] = True
    for hop in range(1, max_hops + 1):
        reached = np.zeros(conn.num_nodes, dtype=bool)
        reached[b[frontier[a]]] = True
        reached &= distance == UNREACHED
        if not reached.any():
            break
        distance[reached] = hop
        frontier = reached
    return distance


def walk_counts(conn: Connectivity, signs, sources, targets, max_hops: int) -> np.ndarray:
    """Return counts[hops-1, sign_class, target] of walks with exactly `hops` synapses.

    Counts are float64 and exact below 2**53; callers must check before casting.
    """
    sources = _validate_nodes(sources, conn.num_nodes, "sources")
    targets = _validate_nodes(targets, conn.num_nodes, "targets")
    signs = np.asarray(signs)
    if signs.shape != (conn.num_nodes,) or not np.isin(signs, (-1, 0, 1)).all():
        raise ValueError("signs must hold -1, 0 or +1 for every node")
    exc, mixed, inh, muted = (np.zeros(conn.num_nodes) for _ in SIGN_CLASSES)
    exc[sources] = 1.0
    pos, neg, zero = signs == 1, signs == -1, signs == 0
    out = np.zeros((max_hops, len(SIGN_CLASSES), len(targets)))
    for hop in range(max_hops):
        leaving = (
            np.where(pos, exc, 0.0),
            np.where(pos, mixed, 0.0) + np.where(neg, inh, 0.0),
            np.where(pos, inh, 0.0) + np.where(neg, exc + mixed, 0.0),
            muted + np.where(zero, exc + mixed + inh, 0.0),
        )
        exc, mixed, inh, muted = (
            np.bincount(conn.dst, weights=v[conn.src], minlength=conn.num_nodes) for v in leaving
        )
        for k, vector in enumerate((exc, mixed, inh, muted)):
            out[hop, k] = vector[targets]
    return out


def sign_class(presynaptic_signs) -> str:
    signs = list(presynaptic_signs)
    if any(s == 0 for s in signs):
        return "muted"
    negatives = sum(s < 0 for s in signs)
    if negatives == 0:
        return "excitatory"
    return "net_inhibitory" if negatives % 2 else "net_excitatory_via_inhibition"


def enumerate_paths(conn: Connectivity, sources, targets, max_hops: int, *, limit: int):
    """Yield every walk of 1..max_hops synapses from a source to a target, in CSR order.

    Each item is (nodes, contacts). A walk may continue through a target to reach
    another target; both are reported. Raises once more than `limit` walks are found.
    """
    sources = _validate_nodes(sources, conn.num_nodes, "sources")
    targets = _validate_nodes(targets, conn.num_nodes, "targets")
    to_target = hop_distances(conn, targets, max_hops, reverse=True)
    is_target = np.zeros(conn.num_nodes, dtype=bool)
    is_target[targets] = True
    found = 0
    for source in sources.tolist():
        stack = [(source, [source], [])]
        while stack:
            node, nodes, contacts = stack.pop()
            used = len(contacts)
            start, end = int(conn.indptr[node]), int(conn.indptr[node + 1])
            children = conn.indices[start:end]
            weights = conn.csr_contacts[start:end]
            budget = max_hops - used - 1
            keep = (to_target[children] != UNREACHED) & (to_target[children] <= budget)
            # Reverse so the stack pops children in CSR order.
            for child, count in zip(children[keep][::-1].tolist(), weights[keep][::-1].tolist()):
                path, synapses = nodes + [child], contacts + [count]
                if is_target[child]:
                    found += 1
                    if found > limit:
                        raise ValueError(f"More than {limit} paths; lower --enumerate-hops")
                    yield path, synapses
                if budget > 0:
                    stack.append((child, path, synapses))


def psp_peak_per_mv(tau_membrane_ms: float, tau_synapse_ms: float) -> float:
    """Peak depolarization per 1 mV jump in g for dv/dt=(v_rest-v+g)/tau_m, dg/dt=-g/tau_s."""
    tm, ts = float(tau_membrane_ms), float(tau_synapse_ms)
    if tm <= 0 or ts <= 0:
        raise ValueError("Time constants must be positive")
    if math.isclose(tm, ts):
        return 1 / math.e
    ratio = ts / tm
    return ts / (tm - ts) * (ratio ** (ts / (tm - ts)) - ratio ** (tm / (tm - ts)))


def direct_drive_estimate(conn: Connectivity, signs, sources, targets, config) -> list[dict]:
    """Linear subthreshold estimate of direct (1-synapse) drive onto each target.

    Uses only the shiu-lif-v1 equations: one presynaptic spike adds w = contacts * sign *
    weight_per_contact_mv to g. Ignores indirect paths, target refractoriness, input
    fluctuations and threshold crossing by variance. A diagnostic, not a prediction.
    """
    sources = _validate_nodes(sources, conn.num_nodes, "sources")
    targets = _validate_nodes(targets, conn.num_nodes, "targets")
    signs = np.asarray(signs)
    is_source = np.zeros(conn.num_nodes, dtype=bool)
    is_source[sources] = True
    gap = config.v_threshold_mv - config.v_rest_mv
    factor = psp_peak_per_mv(config.tau_membrane_ms, config.tau_synapse_ms)
    tau_s = config.tau_synapse_ms / 1000
    rows = []
    for target in targets.tolist():
        mask = is_source[conn.src] & (conn.dst == target)
        presyn = conn.src[mask]
        contacts = conn.contacts[mask].astype(np.int64)
        weight = float(np.sum(contacts * signs[presyn]) * config.weight_per_contact_mv)
        rate = gap / (weight * tau_s) if weight > 0 else None
        rows.append(
            {
                "target_node": int(target),
                "direct_sources": int(len(np.unique(presyn))),
                "direct_contacts": int(contacts.sum()),
                "direct_weight_mv": weight,
                "synchronous_volley_peak_mv": weight * factor,
                "threshold_gap_mv": gap,
                "uniform_source_rate_for_mean_threshold_hz": rate,
            }
        )
    return rows
