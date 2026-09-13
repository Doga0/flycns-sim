# MaleCNS Graph Core — v0.2

This layer answers which released segments are retained, which retained segments
connect, and how to query those connections efficiently. It does not connect the
CNS to FlyGym or assign neural dynamics to the graph.

## 1. Audit the census first

From the repository root, with the three raw Feather files already downloaded:

```powershell
uv run python -m malecns_sim.cns.census --output outputs/census.json
```

Docker equivalent, after `docker compose build`:

```powershell
docker compose run --rm sim python -m malecns_sim.cns.census --output outputs/census.json
```

The terminal shows headline counts and status/superclass counts. The JSON also
contains all observed combinations of status, superclass, class, subclass, type,
and available side fields; per-column labels; and policy differences. Missing
values appear as JSON `null`, distinct from an empty string. No missing side is
inferred: the release has `somaSide` and `rootSide`, not a single `side` column.

The [publication reports 166,691 neurons](https://research.google/pubs/sexual-dimorphism-in-the-complete-connectome-of-the-drosophila-male-central-nervous-system/).
The actual downloaded annotations produce:


| Audit item                            |   Count |
| --------------------------------------- | --------: |
| Total annotations                     | 211,577 |
| `status == Traced`                    | 165,122 |
| `status == Glia`                      |  11,864 |
| Assigned, nonempty superclass         | 166,700 |
| Traced without an assigned superclass |     516 |
| Assigned superclass but not Traced    |   2,094 |

The 2,094 additional assigned-superclass entries consist of 2,002 with missing
status, 60 Anchors, and 32 Orphans. Thus the superclass policy is not a superset
of the Traced policy: it also excludes the 516 Traced entries without superclass.

## 2. Choose an explicit node policy


| Profile         | File                               | Selection                                                |   Nodes | Difference from publication |
| ----------------- | ------------------------------------ | ---------------------------------------------------------- | --------: | ----------------------------: |
| `published`     | `configs/graph/published.yaml`     | Assigned superclass, excluding explicit Glia             | 166,700 |                          +9 |
| `traced`        | `configs/graph/traced.yaml`        | Exactly`status == Traced`                                | 165,122 |                     −1,569 |
| `all-annotated` | `configs/graph/all-annotated.yaml` | Every annotation, including glia and unresolved segments | 211,577 |                     +44,886 |

**`published-v1` is a provisional census approximation.** Count proximity does
not establish membership equivalence. We have not identified an authoritative
versioned body-ID list or publication-specific rule that explains the remaining
nine-node difference. No nine nodes are arbitrarily removed to force a match.
`reference_membership_verified: false` and the signed count difference are
recorded in both the census and manifest. The all-annotated profile is a segment
graph, not a claim that all annotation rows are neurons.

This choice is consistent with the assigned-superclass approach documented in
[Doomfly's importer](https://github.com/nftechie/doomfly/blob/main/doom/connectome.py).
[Flyhard's pilot report](https://github.com/MarkUnthank/flyhard/blob/main/docs/pilot-2026-09-09.md)
provides a separate Traced comparison. External counts are context, not targets
hard-coded into the builder.

Policies are YAML data, separate from graph code. Rules support `all`, `any`,
`not`, and leaves containing `column` plus either `in` or `assigned`. Unknown
fields and missing columns fail explicitly. There is no expression evaluation.
`assigned` means neither null nor empty/whitespace-only. A custom profile can be
passed as `--policy configs/graph/my-profile.yaml`; use a new versioned `id` when
changing scientific membership. Built-in YAML files are included in the Python
wheel and Docker image as well as the repository.

All v0.2 profiles preserve self-edges and all positive released contact counts
between retained nodes. Edge thresholds and other edge policies are rejected.

## 3. Build the graph

Local Windows:

```powershell
uv sync --locked
uv run python -m malecns_sim.cns.graph.build --policy published
```

Docker:

```powershell
docker compose build
docker compose run --rm sim python -m malecns_sim.cns.graph.build --policy published
```

Choose one workflow for the initial build. Both write into the same host data
directory. Raw files remain unchanged. The output is:

```text
data/processed/malecns-v1.0/published-v1/
├── nodes.parquet
├── body_ids.npy
├── src.npy
├── dst.npy
├── contacts.npy
├── csr_indptr.npy
├── csr_indices.npy
├── csr_contacts.npy
├── census.json
├── audit.json
└── manifest.json
```

If the destination exists, the builder fails instead of overwriting it. To build
another copy, use `--output data/processed/malecns-v1.0/published-v1-repeat`.
Files are assembled in a temporary sibling directory and published by directory
rename only after validation. Failed builds do not appear as completed graphs.

### Memory and disk behavior

The annotation and neurotransmitter tables are small enough to align in pandas.
The 151,856,684-row connectivity table is **never loaded into a pandas DataFrame**.
The builder projects only `body_pre`, `body_post`, and `weight` using PyArrow IPC
record batches (Feather v2), maps endpoints to retained IDs, and accounts for every
excluded row and contact. Extra input columns are not read into the working batch.

Retained records are partitioned on disk by source-node range. Each partition is
sorted and aggregated in bounded chunks. Oversized partitions use external sorted
runs and a merge with at most 32 input streams. COO files are filled through NumPy
memory maps, and CSR is built from outgoing degrees. No NetworkX objects or dense
N×N matrix are created.

Defaults are `--batch-rows 65536 --chunk-rows 1000000 --partitions 64`. Chunk sizes
bound working arrays, not the OS file cache. Arrow must still decompress one
upstream record batch at a time; the real input uses batches of at most 65,536
rows, recorded in the audit. Reducing chunk size lowers sort memory but may make
external merging slower. Temporary disk space grows with retained connectivity;
allow about 2 GB of free working space for one default published build, beyond
the raw data and installed environment. Other profiles can require more space.

## 4. Understand the result and aggregation audit

The actual Windows published build produced:


| Item                               |      Result |
| ------------------------------------ | ------------: |
| Retained nodes                     |     166,700 |
| Raw connectivity rows              | 151,856,684 |
| Retained connectivity rows         |  25,582,938 |
| Unique directed retained pairs     |  25,582,938 |
| Retained duplicate rows aggregated |           0 |
| Raw contacts                       | 311,833,243 |
| Retained contacts                  | 124,177,617 |
| Excluded contacts                  | 187,655,626 |

The [official release](https://male-cns.janelia.org/download/) describes this file
as the full segment-to-segment connection graph. Its actual columns contain pair
weights, not individual synapse coordinates or ROI labels. **No duplicate retained
pairs were found.** For this input and policy, the reduction from 151 million to
25.6 million rows comes from endpoint selection, not merging duplicate retained
rows. The builder nevertheless sums duplicates correctly, including duplicates
across batches and disk runs, as tested with synthetic fixtures.

The full raw file is not globally lexicographically sorted. Its adjacent duplicate
count is zero, but that does not prove global uniqueness outside the retained
graph. Accordingly, `raw_duplicate_rows` is `null`, not a guessed zero. If duplicates
occur in another input, the three columns alone cannot establish their upstream
cause; the audit explicitly says so.

Exclusions use disjoint endpoint categories:


| Exclusion            |        Rows |    Contacts |
| ---------------------- | ------------: | ------------: |
| Only source excluded |   4,802,752 |   6,276,306 |
| Only target excluded | 112,633,588 | 170,891,397 |
| Both excluded        |   8,837,406 |  10,487,923 |
| Total excluded       | 126,273,746 | 187,655,626 |

`raw_contacts = retained_contacts + excluded_contacts`, and aggregated graph
contacts equal retained contacts exactly. Excluded IDs include any endpoints not
retained by the selected annotation policy; they are not automatically all glia.

### Canonical arrays and metadata


| File               | Type / shape               | Meaning                            |
| -------------------- | ---------------------------- | ------------------------------------ |
| `body_ids.npy`     | little-endian uint64`[N]`  | Original body IDs sorted ascending |
| `src.npy`          | little-endian int32`[E]`   | Source node index                  |
| `dst.npy`          | little-endian int32`[E]`   | Target node index                  |
| `contacts.npy`     | little-endian uint32`[E]`  | Positive summed synaptic contacts  |
| `csr_indptr.npy`   | little-endian int64`[N+1]` | Outgoing edge offsets              |
| `csr_indices.npy`  | little-endian int32`[E]`   | Target indices in outgoing order   |
| `csr_contacts.npy` | little-endian uint32`[E]`  | Corresponding contact counts       |

Edges are unique and sorted lexicographically by `(src, dst)`. Isolated retained
nodes stay in the graph. IDs are never converted through floating point. Negative
or noninteger IDs, null/nonpositive/noninteger weights, duplicate metadata IDs,
and uint32 aggregation overflow fail explicitly rather than being silently fixed.

`nodes.parquet` has contiguous `node_index` values and `body_id`, followed by all
released annotation fields and aligned neurotransmitter metadata. It preserves
the distinct confidence fields and `somaSide`/`rootSide`. Missing NT rows remain
missing. No generic consensus confidence, inferred side, excitatory/inhibitory
sign, receptor property, or neural weight is invented.

Canonical COO, body IDs, and node metadata occupy about **316 MB**. Including
the explicit CSR cache and audit reports, the complete directory is about
**529 MB** (decimal MB). CSR duplicates the target/contact arrays intentionally
to provide the requested portable cache files.

### Provenance and deterministic output

The manifest records the dataset and graph schema, builder version, full policy,
publication discrepancy, input filenames/URLs/byte sizes/MD5/SHA-256 hashes,
computed node/edge/contact counts, exclusion accounting, ordering, dtypes, and
output artifact checksums. It contains no build timestamp, machine name, absolute
input path, or random run ID. JSON uses sorted keys and LF newlines; arrays use
fixed little-endian dtypes. Use the committed dependency lock for reproducibility.

## 5. Load, query, and visualize the graph

The graph can be queried either through the Python API in `store.py` or through the convenience CLI in `query.py`. The CLI is intended for inspection and debugging only; it does not modify the graph artifacts or add neural dynamics.

### Python API

In Python within the project environment:

```python
from malecns_sim.cns.graph.store import CNSGraph

with CNSGraph.load("data/processed/malecns-v1.0/published-v1") as graph:
    index = 0
    original_body_id = int(graph.body_ids[index])

    assert graph.node_index(original_body_id) == index

    targets, contacts = graph.outgoing(index)

    print(original_body_id, targets[:10], contacts[:10])
```

`outgoing()` takes an internal node index, not a MaleCNS body ID. It returns read-only memory-mapped slices using CSR offsets. Use the slices while the graph is open, or copy them before leaving the context. An isolated node returns empty arrays. An unknown body ID raises `KeyError`; an invalid node index raises `IndexError`. Loading does not scan all edge files; explicit validation does.

### `query.py` command-line inspection

`query.py` provides a human-readable view of a selected neuron and its local 1-hop connectivity. Run it from the repository root after the graph has been built.

Query by internal node index:
```powershell
uv run python -m malecns_sim.cns.graph.query --node 1204
```

Query the same neuron by its original MaleCNS body ID instead:
```powershell
uv run python -m malecns_sim.cns.graph.query --body-id 11296
```

The two identifiers have different meanings:
```text
N1204 / node_index 1204   -> internal contiguous graph index
body_id 11296            -> original MaleCNS released segment/neuron identifier
```

The default text view reports the center neuron, its total outgoing degree, and a small set of the strongest incoming and outgoing connections. Edge numbers are released contact counts. For example:

```text
Incoming:

    N1357 --148--> [ N1204 ]
    N2283 --82---> [ N1204 ]

[ N1204 ]
    |--219--> N253
    |--78---> N10850
    `--66---> N8788
```

This means that the selected retained node receives a released directed connection from `N1357` with weight/contact count 148, and sends a connection to `N253` with weight/contact count 219. These values are anatomical graph weights; v0.2 does not interpret them as membrane voltage, firing rate, excitatory or inhibitory effect, or motor output.

Show the center node's released annotation and aligned neurotransmitter metadata:
```powershell
uv run python -m malecns_sim.cns.graph.query --node 1204 --metadata
```

Control how many strong neighbors are displayed:
```powershell
uv run python -m malecns_sim.cns.graph.query --node 1204 --incoming 10 --outgoing 15
```

The displayed neighbors are a visualization subset, not the complete neighborhood. For example, `total out-degree=890` means that the center node has 890 distinct outgoing target nodes in the retained graph even when only the top 8 or 15 are drawn.

### Plot the 1-hop neighborhood

If plotting support is not already installed in the environment:
```powershell
uv add matplotlib
```

Open an interactive plot window:
```powershell
uv run python -m malecns_sim.cns.graph.query --node 1204 --plot
```

The plot places the selected neuron in the center, strongest incoming neighbors on the left, and strongest outgoing neighbors on the right. Node labels use the internal graph index and, when available, the released `type` metadata. Numbers on arrows are contact counts. Line thickness is only a visual aid; use the printed edge number as the actual contact count.

Choose the number of displayed neighbors:
```powershell
uv run python -m malecns_sim.cns.graph.query --node 1204 --incoming 5 --outgoing 8 --plot
```

Save the same plot without opening a window:
```powershell
uv run python -m malecns_sim.cns.graph.query --node 1204 --save-plot outputs/n1204.png
```

Save and display it at the same time:
```powershell
uv run python -m malecns_sim.cns.graph.query --node 1204 --plot --save-plot outputs/n1204.png
```

Suppress the terminal neighborhood and show only the plot:
```powershell
uv run python -m malecns_sim.cns.graph.query --node 1204 --plot --no-text
```

The plot is deliberately limited to a small 1-hop subset. The complete graph has tens of millions of directed edges and is not copied into NetworkX or a dense matrix for visualization. Outgoing neighbors are obtained directly from the CSR cache. Because v0.2 stores only outgoing CSR, `query.py` finds incoming neighbors by scanning the memory-mapped `dst.npy` array in bounded chunks.`query.py` is an inspection tool only. It does not change `nodes.parquet`, the COO/CSR arrays, the manifest, node policy, contact counts, or the v0.2 scientific boundary.

## 6. Validate and compare builds

```powershell
uv run python -m malecns_sim.cns.graph.validate data/processed/malecns-v1.0/published-v1 --output-report outputs/graph-validation-windows.json
uv run pytest tests/test_graph.py tests/test_cns.py -q
```

This validates mapping bijection, index bounds, positive contacts, sorted unique
pairs, CSR/COO agreement, metadata order and IDs, contact conservation, self-edge
counts, checksums, and outgoing queries for 100 reproducibly sampled nodes.
The real-data acceptance test also compares retained annotation metadata against
the original annotation file. Graph tests do not run the physical simulation.

Small fixtures only, without raw data or a previously built graph:

```powershell
uv run pytest tests/test_graph.py -m "not graph" -q
```

To compare genuinely independent Windows and Docker builds, first build locally,
then use a different output directory in Docker:

```powershell
docker compose run --rm sim python -m malecns_sim.cns.graph.build --policy published --output data/processed/malecns-v1.0/published-v1-docker
docker compose run --rm sim python -m malecns_sim.cns.graph.validate data/processed/malecns-v1.0/published-v1-docker --compare data/processed/malecns-v1.0/published-v1 --output-report outputs/graph-validation-docker.json
docker compose run --rm sim pytest tests/test_graph.py tests/test_cns.py -q
```

`--compare` checks both graphs independently and requires identical summaries,
artifact hashes, and manifest hashes. It fails on any difference. The fixture
suite also builds the same input with different batch/chunk/partition sizes and
compares every output file byte-for-byte. No reference edge count is forced.

The unqualified `pytest` command now includes the graph acceptance test and needs
both the raw files and the built `published-v1` graph. To run only the original
v0.1 checks, explicitly select `tests/test_cns.py tests/test_simulation.py`.

## Boundary after v0.2

Stop at anatomical graph storage and queries. Neural dynamics, neurotransmitter
signs, sensory/motor catalogues, CNS-to-actuator mapping, feedback, learning, and
RL belong to later releases. The existing simulation joint control is unchanged.
