"""Stream released Feather connectivity into deterministic COO and outgoing CSR."""

import argparse
import heapq
import math
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from malecns_sim import __version__
from malecns_sim.cns.census import audit_annotations
from malecns_sim.cns.download import BASE_URL, DEFAULT_DATA_DIR, FILES
from malecns_sim.cns.graph.policy import exact_ids, load_policy, select_nodes
from malecns_sim.cns.graph.store import ARRAY_DTYPES, CNSGraph, file_hashes, write_json

RECORD = np.dtype([("key", "<u8"), ("contacts", "<u8")])
UINT32_MAX = int(np.iinfo(np.uint32).max)
UINT64_MAX = int(np.iinfo(np.uint64).max)
BUFFER_ROWS = 8192


def read_batches(path: Path, columns: list[str], batch_rows: int):
    """Project only required IPC fields; bound working slices to batch_rows.

    Feather v2 stores IPC record batches. Decompression still needs one source
    record batch at a time; that upstream batch size is reported in the audit.
    """
    with pa.memory_map(str(path), "r") as source:
        schema = ipc.open_file(source).schema
        missing = set(columns) - set(schema.names)
        if missing:
            raise ValueError(f"Missing Feather columns: {sorted(missing)}")
        options = ipc.IpcReadOptions(included_fields=[schema.get_field_index(c) for c in columns])
        reader = ipc.open_file(source, options=options)
        for number in range(reader.num_record_batches):
            batch = reader.get_batch(number)
            for offset in range(0, batch.num_rows, batch_rows):
                yield batch.slice(offset, batch_rows), batch.num_rows


def make_nodes(raw_dir: Path, policy: dict) -> tuple[pa.Table, dict]:
    annotations = pd.read_feather(raw_dir / FILES["annotations"])
    annotation_ids = exact_ids(annotations.bodyId.to_numpy())
    if len(np.unique(annotation_ids)) != len(annotation_ids):
        raise ValueError("Duplicate annotation body IDs")
    selected = annotations.loc[select_nodes(annotations, policy["selection"])].copy()
    selected = selected.rename(columns={"bodyId": "body_id"})
    selected["body_id"] = exact_ids(selected.body_id.to_numpy())
    selected = selected.sort_values("body_id", kind="stable").reset_index(drop=True)
    if len(selected) == 0 or len(selected) > np.iinfo(np.int32).max:
        raise ValueError("Policy must retain 1..INT32_MAX nodes")
    selected.insert(0, "node_index", np.arange(len(selected), dtype="<i4"))
    nt = pd.read_feather(raw_dir / FILES["neurotransmitters"])
    nt["body"] = exact_ids(nt.body.to_numpy(), "NT body")
    if nt.body.duplicated().any():
        raise ValueError("Duplicate neurotransmitter body IDs would multiply node metadata")
    if (set(selected) & set(nt)) - {"body"}:
        raise ValueError("Annotation and neurotransmitter metadata column collision")
    nt = nt.set_index("body")
    # Reindex only the small retained node table; preserve unknown NT values as null.
    aligned = nt.reindex(selected.body_id).reset_index(drop=True)
    nodes = pd.concat([selected, aligned], axis=1)
    table = pa.Table.from_pandas(nodes, preserve_index=False).replace_schema_metadata(None)
    audit = audit_annotations(annotations, [policy])
    audit["metadata"] = {
        "nt_rows": len(nt),
        "nodes_without_nt_row": int((~selected.body_id.isin(nt.index)).sum()),
        "side": "Original somaSide/rootSide fields retained; no inferred side column.",
        "confidence": (
            "Original predicted_nt_confidence and celltype_predicted_nt_confidence retained; "
            "no consensus confidence invented."
        ),
        "transmitter_signs": "Not assigned; all released neurotransmitter labels remain metadata.",
    }
    return table, audit


def aggregate_records(records: np.ndarray) -> np.ndarray:
    if not len(records):
        return records
    records.sort(order="key")
    starts = np.r_[0, np.flatnonzero(records["key"][1:] != records["key"][:-1]) + 1]
    totals = np.add.reduceat(records["contacts"], starts)
    if np.any(totals > UINT32_MAX):
        raise OverflowError("Aggregated pair contacts exceed uint32")
    result = np.empty(len(starts), dtype=RECORD)
    result["key"] = records["key"][starts]
    result["contacts"] = totals
    return result


def record_iterator(path: Path):
    with path.open("rb") as stream:
        while len(chunk := np.fromfile(stream, dtype=RECORD, count=BUFFER_ROWS)):
            for row in chunk:
                yield int(row["key"]), int(row["contacts"])


def merge_runs(paths: list[Path], destination: Path) -> None:
    """Bounded fan-in merge; cross-chunk duplicates sum as Python integers."""
    buffer = np.empty(BUFFER_ROWS, dtype=RECORD)
    used = 0
    current_key, total = None, 0
    with destination.open("wb") as stream:
        for key, weight in heapq.merge(*(record_iterator(path) for path in paths)):
            if key != current_key and current_key is not None:
                buffer[used] = (current_key, total)
                used += 1
                if used == len(buffer):
                    buffer.tofile(stream)
                    used = 0
                total = 0
            current_key = key
            total += weight
            if total > UINT32_MAX:
                raise OverflowError("Aggregated pair contacts exceed uint32")
        if current_key is not None:
            buffer[used] = (current_key, total)
            used += 1
        buffer[:used].tofile(stream)


def sort_partition(path: Path, chunk_rows: int) -> Path:
    runs = []
    with path.open("rb") as stream:
        while len(chunk := np.fromfile(stream, dtype=RECORD, count=chunk_rows)):
            run = path.with_name(f"{path.stem}-run-{len(runs)}.bin")
            aggregate_records(chunk).tofile(run)
            runs.append(run)
    if not runs:
        return path
    level = 0
    while len(runs) > 1:
        merged = []
        for offset in range(0, len(runs), 32):
            group = runs[offset : offset + 32]
            destination = path.with_name(f"{path.stem}-merge-{level}-{offset}.bin")
            merge_runs(group, destination)
            for run in group:
                run.unlink()
            merged.append(destination)
        runs = merged
        level += 1
    path.unlink()
    return runs[0]


def scan_connections(path: Path, ids: np.ndarray, scratch: Path, batch_rows: int, partitions: int):
    buckets = [scratch / f"partition-{i:04d}.bin" for i in range(partitions)]
    stats = {
        "raw_rows": 0,
        "raw_contacts": 0,
        "retained_rows": 0,
        "retained_contacts": 0,
        "excluded_rows": 0,
        "excluded_contacts": 0,
        "excluded_source_only_rows": 0,
        "excluded_source_only_contacts": 0,
        "excluded_target_only_rows": 0,
        "excluded_target_only_contacts": 0,
        "excluded_both_rows": 0,
        "excluded_both_contacts": 0,
        "raw_lexicographically_sorted": True,
        "raw_adjacent_duplicate_rows": 0,
        "source_max_record_batch_rows": 0,
    }
    previous = None
    nodes_per_bucket = math.ceil(len(ids) / partitions)
    with ExitStack() as stack:
        streams = [stack.enter_context(p.open("wb")) for p in buckets]
        for batch, source_rows in read_batches(
            path, ["body_pre", "body_post", "weight"], batch_rows
        ):
            if any(batch.column(c).null_count for c in ("body_pre", "body_post", "weight")):
                raise ValueError("Connectivity contains null IDs or weights")
            pre, post = [
                exact_ids(batch.column(c).to_numpy(), c) for c in ("body_pre", "body_post")
            ]
            weights = batch.column("weight").to_numpy()
            if (
                weights.dtype.kind not in "iu"
                or np.any(weights <= 0)
                or np.any(weights > UINT32_MAX)
            ):
                raise ValueError("Raw contact counts must be positive uint32-compatible integers")
            if not len(pre):
                continue
            weights = weights.astype("<u8", copy=False)
            stats["source_max_record_batch_rows"] = max(
                stats["source_max_record_batch_rows"], source_rows
            )
            first = (int(pre[0]), int(post[0]))
            if previous is not None:
                stats["raw_lexicographically_sorted"] &= first >= previous
                stats["raw_adjacent_duplicate_rows"] += int(first == previous)
            previous = (int(pre[-1]), int(post[-1]))
            equal_pre = pre[1:] == pre[:-1]
            stats["raw_lexicographically_sorted"] &= bool(
                np.all((pre[1:] > pre[:-1]) | (equal_pre & (post[1:] >= post[:-1])))
            )
            stats["raw_adjacent_duplicate_rows"] += int(
                np.count_nonzero(equal_pre & (post[1:] == post[:-1]))
            )
            src, dst = np.searchsorted(ids, pre), np.searchsorted(ids, post)
            known_src = (src < len(ids)) & (ids[np.minimum(src, len(ids) - 1)] == pre)
            known_dst = (dst < len(ids)) & (ids[np.minimum(dst, len(ids) - 1)] == post)
            keep = known_src & known_dst
            masks = {
                "raw": np.ones(len(pre), dtype=bool),
                "retained": keep,
                "excluded": ~keep,
                "excluded_source_only": ~known_src & known_dst,
                "excluded_target_only": known_src & ~known_dst,
                "excluded_both": ~known_src & ~known_dst,
            }
            for name, mask in masks.items():
                stats[f"{name}_rows"] += int(np.count_nonzero(mask))
                stats[f"{name}_contacts"] += int(weights[mask].sum(dtype=np.uint64))
            if stats["raw_contacts"] > UINT64_MAX:
                raise OverflowError("Raw contact total exceeds uint64")
            records = np.empty(int(keep.sum()), dtype=RECORD)
            records["key"] = (src[keep].astype("<u8") << np.uint64(32)) | dst[keep].astype("<u8")
            records["contacts"] = weights[keep]
            bucket_ids = src[keep] // nodes_per_bucket
            # Group once rather than scanning every retained row once per partition.
            order = np.argsort(bucket_ids, kind="stable")
            records, bucket_ids = records[order], bucket_ids[order]
            starts = np.r_[0, np.flatnonzero(bucket_ids[1:] != bucket_ids[:-1]) + 1, len(records)]
            for left, right in zip(starts[:-1], starts[1:]):
                if left != right:
                    records[left:right].tofile(streams[int(bucket_ids[left])])
    return buckets, stats


def write_arrays(directory: Path, nodes: pa.Table, runs: list[Path], chunk_rows: int) -> dict:
    n = nodes.num_rows
    edges = sum(path.stat().st_size // RECORD.itemsize for path in runs)
    arrays = {}
    for name, dtype in ARRAY_DTYPES.items():
        size = n if name == "body_ids" else n + 1 if name == "csr_indptr" else edges
        arrays[name] = np.lib.format.open_memmap(
            directory / f"{name}.npy", mode="w+", dtype=dtype, shape=(size,)
        )
    degree = np.zeros(n, dtype="<i8")
    arrays["body_ids"][:] = nodes["body_id"].to_numpy()
    offset, contacts, self_edges = 0, 0, 0
    try:
        for run in runs:
            with run.open("rb") as stream:
                while len(chunk := np.fromfile(stream, dtype=RECORD, count=chunk_rows)):
                    src = (chunk["key"] >> np.uint64(32)).astype("<i4")
                    dst = (chunk["key"] & np.uint64(UINT32_MAX)).astype("<i4")
                    end = offset + len(chunk)
                    for name, values in (
                        ("src", src),
                        ("dst", dst),
                        ("contacts", chunk["contacts"]),
                        ("csr_indices", dst),
                        ("csr_contacts", chunk["contacts"]),
                    ):
                        arrays[name][offset:end] = values
                    degree += np.bincount(src, minlength=n)
                    contacts += int(chunk["contacts"].sum(dtype=np.uint64))
                    self_edges += int(np.count_nonzero(src == dst))
                    offset = end
        arrays["csr_indptr"][0] = 0
        np.cumsum(degree, out=arrays["csr_indptr"][1:])
    finally:
        for array in arrays.values():
            array.flush()
            array._mmap.close()
    return {"nodes": n, "edges": edges, "contacts": contacts, "self_edges": self_edges}


def build_graph(
    raw_dir: Path,
    output: Path,
    policy: dict,
    *,
    batch_rows: int = 65536,
    chunk_rows: int = 1_000_000,
    partitions: int = 64,
) -> dict:
    if batch_rows < 1 or chunk_rows < 1 or not 1 <= partitions <= 256:
        raise ValueError("Positive batch/chunk sizes and 1..256 partitions required")
    if max(batch_rows, chunk_rows) > UINT64_MAX // UINT32_MAX:
        raise ValueError("Batch size could overflow contact accumulation")
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    inputs = {
        name: {"filename": filename, "url": BASE_URL + filename, **file_hashes(raw_dir / filename)}
        for name, filename in FILES.items()
    }
    print("Auditing node membership and aligning metadata...", flush=True)
    nodes, census = make_nodes(raw_dir, policy)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp:
        stage = Path(temp) / "graph"
        scratch = Path(temp) / "scratch"
        stage.mkdir()
        scratch.mkdir()
        pq.write_table(
            nodes,
            stage / "nodes.parquet",
            compression="zstd",
            version="2.6",
            row_group_size=65536,
            use_dictionary=True,
            write_statistics=True,
        )
        print("Streaming connectivity and accounting for excluded contacts...", flush=True)
        buckets, stats = scan_connections(
            raw_dir / FILES["connections"],
            nodes["body_id"].to_numpy(),
            scratch,
            batch_rows,
            partitions,
        )
        runs = []
        for number, bucket in enumerate(buckets):
            runs.append(sort_partition(bucket, chunk_rows))
            if (number + 1) % 16 == 0:
                print(f"Aggregated {number + 1}/{len(buckets)} partitions", flush=True)
        summary = write_arrays(stage, nodes, runs, chunk_rows)
        if summary["contacts"] != stats["retained_contacts"]:
            raise ValueError("Contact conservation failed")
        stats["retained_duplicate_rows_aggregated"] = stats["retained_rows"] - summary["edges"]
        stats["raw_duplicate_rows"] = (
            stats["raw_adjacent_duplicate_rows"] if stats["raw_lexicographically_sorted"] else None
        )
        stats["aggregation_interpretation"] = (
            "Released fields are body_pre, body_post, weight: segment-pair contact counts, "
            "not individual synapse coordinates or ROI rows. Duplicate retained pairs are "
            "summed without thresholds. When the full input is sorted, adjacent duplicates "
            "give the exact raw duplicate count; otherwise global raw duplication is unknown. "
            "If duplicate pairs occur, these columns alone cannot establish their upstream cause."
        )
        write_json(stage / "census.json", census)
        write_json(stage / "audit.json", stats)
        artifact_names = ["nodes.parquet", "census.json", "audit.json"] + [
            f"{n}.npy" for n in ARRAY_DTYPES
        ]
        manifest = {
            "dataset": "MaleCNS v1.0",
            "graph_schema": 1,
            "builder_version": __version__,
            "node_policy": policy["id"],
            "policy": policy,
            "edge_policy": "all-released",
            "self_edges_preserved": True,
            "edge_threshold": None,
            "reference_neurons": policy["reference_neurons"],
            "difference_from_reference": nodes.num_rows - policy["reference_neurons"],
            "reference_membership_verified": policy["reference_membership_verified"],
            "ordering": "body_id ascending; unique edges lexicographic (src, dst)",
            "dtypes": ARRAY_DTYPES,
            "input_files": inputs,
            "artifacts": {name: file_hashes(stage / name) for name in sorted(artifact_names)},
            "contact_accounting": {
                k: v for k, v in stats.items() if k.endswith(("_rows", "_contacts", "_aggregated"))
            },
            **summary,
        }
        write_json(stage / "manifest.json", manifest)
        from malecns_sim.cns.graph.validate import validate_graph

        with CNSGraph.load(stage) as graph:
            validate_graph(graph)
        for name, filename in FILES.items():
            current = file_hashes(raw_dir / filename)
            if any(current[key] != inputs[name][key] for key in current):
                raise ValueError(f"Input changed during build: {filename}")
        # Publish only after all checks; never overwrite an existing graph.
        stage.rename(output)
    print(f"Graph ready: {output}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--policy", default="published", help="Built-in profile or YAML path")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-rows", type=int, default=65536)
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    parser.add_argument("--partitions", type=int, default=64)
    args = parser.parse_args()
    policy = load_policy(args.policy)
    output = args.output or Path("data/processed/malecns-v1.0") / policy["id"]
    result = build_graph(
        args.data_dir,
        output,
        policy,
        batch_rows=args.batch_rows,
        chunk_rows=args.chunk_rows,
        partitions=args.partitions,
    )
    print({k: result[k] for k in ("nodes", "edges", "contacts", "difference_from_reference")})


if __name__ == "__main__":
    main()
