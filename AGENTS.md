# AGENTS.md: shared instructions for coding agents

`flycns-sim` (Python package `malecns_sim`) simulates the MaleCNS v1.0 *Drosophila*
connectome as a Brian2 LIF network and couples it to a FlyGym 2.1 / MuJoCo 3.9 body.
Read this file fully before changing anything.

## 1. Before you implement anything

1. Read [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md) (where development stands, the
   next task, open questions) and the parts of [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
   that cover the modules you will touch. [docs/DECISIONS.md](docs/DECISIONS.md) explains
   constraints that look odd but are deliberate.
2. Verify the docs against reality. Run `git status`, `git log --oneline -10` and
   `git diff --stat`, and open the source files the docs point to. The docs can be stale;
   the code and git history win. If they disagree, fix the doc in the same change.
3. Read the feature guide for the layer you touch (`docs/graph-core.md`,
   `docs/neural-reference.md`, `docs/io-catalogue.md`, `docs/motor-bridge.md`,
   `docs/sensory-bridge.md`, `docs/closed-loop.md`). They hold exact commands, schemas and
   measured results.
4. Explore before editing: follow imports from the entry point (`python -m malecns_sim.<…>`)
   down to the code you change, and read the existing tests for it.

## 2. Hard rules

- **Preserve uncommitted work.** Never `git reset`, `git checkout -- <file>`, `git clean`,
  `git stash drop` or overwrite files you did not write. Check `git status` first. Do not
  commit, push or open a PR unless the user asks.
- **Never overwrite data or outputs.** `data/` (about 1.1 GB raw MaleCNS tables plus built
  artifacts) and `outputs/` are git-ignored and are not reproducible cheaply. Builders and
  experiments refuse existing output directories by design (`bridge/store.py:stage_output`,
  `FileExistsError` checks). Keep it that way, and always pass a *new* `--output`.
- **No silent scientific shortcuts.** This codebase consistently refuses to:
  infer neuron identity from type-name substrings, change weights or LIF parameters to get
  a result, choose a "lucky" seed, inject artificial motor spikes into real-data runs, or
  relabel other motor cells to make an acceptance gate pass. New mappings use explicit,
  reviewed body IDs in a versioned YAML profile. See DECISIONS.md D5–D8.
- **Label assumptions.** Engineering conventions (decoder gains, encoder transfer functions,
  actuator bounds, safety guards) are labeled as such in configs and reports. Do not
  describe them as biological facts.
- **Keep layer boundaries.** `cns/` never assigns signs or dynamics; `neural/` never touches
  MuJoCo; `simulation/` never reads CNS data; only `bridge/` couples them.
- **Versioned profiles are immutable once results depend on them.** Changing the behavior of
  `configs/*/…_v1.yaml` invalidates recorded hashes (`expected_lif_config_sha256`, mapping
  manifests). Add a `_v2` profile instead.
- Focused changes over rewrites. Match the surrounding style: short docstrings stating the
  boundary, explicit `ValueError` validation, provenance and hashes on every artifact.
- Do not expose secrets. The project needs none; the only network access is the public
  MaleCNS download and dependency installation.

## 3. Essential commands

Run from the repository root. Shell examples are PowerShell (the dev machine is Windows 11);
they work unchanged in bash except for line continuations.

```powershell
uv sync --locked                                   # Python 3.12 env from uv.lock
uv run pytest -q                                   # full suite (needs data/ and built artifacts; slow)
uv run pytest -q -m "not dataset and not graph and not io_catalogue"   # no-data subset
uv run pytest tests/neural/test_incremental.py tests/bridge/test_closed_loop.py -q
uv run ruff check .
uv run ruff format --check .
docker compose build                               # headless Linux image (OSMesa)
docker compose run --rm sim pytest -q
```

Pipeline order when artifacts are missing (each step refuses to overwrite):
`cns.download` → `cns.census` → `cns.graph.build --policy published` → `io.build` →
`bridge.build` → `bridge.sensory.build` → `bridge.closed_loop.runner --output …`.
Details and validation commands are in the feature guides and README.

Notes:
- Pytest markers `dataset`, `graph`, `io_catalogue` need the raw files / built
  `published-v1` graph / `io-v1` catalogue. Some real-data tests fail (not skip) when data
  is missing, by design.
- If you use `uv run` while `uv.lock` has uncommitted changes, prefer `uv run --frozen` so
  the lock file is not rewritten.
- Full-CNS runs use 166,700 neurons and 25.6 M edges; they take minutes and several GB of
  RAM. Start with the toy or subgraph experiments when testing neural changes.

## 4. Testing and verification

- Run the tests for every module you touch, plus `ruff check`. Run the full suite before
  calling a substantial change done if data is available.
- Report actual results: the exact command, pass/fail/skip counts, and failures verbatim.
  Never claim a test, experiment or Docker comparison ran if it did not.
- Scientific acceptance is separate from tests. A green suite does not mean a milestone
  passed; check the relevant `validate` command (for v0.7:
  `bridge.closed_loop.validate <run> --require-effect`).
- For changes to the neural backend or decoder, keep Windows/Docker determinism: the
  existing `--compare` validators require exact discrete events.

## 5. Maintaining documentation

After any substantial change (new feature, behavior change, new result, changed command):

1. Update [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md): stage, what is verified, test
   status (with date and command), git state, next task.
2. Update [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) if modules, interfaces or data flow
   changed.
3. Add an entry to [docs/DECISIONS.md](docs/DECISIONS.md) for any meaningful architectural
   or scientific-policy decision, with evidence and trade-offs. Mark unknown rationale as
   unknown; never invent history.
4. Tick or revise [docs/ROADMAP.md](docs/ROADMAP.md).
5. Keep the feature guide for the touched layer accurate (commands, schemas, measured
   numbers).

Keep docs concise and link to source files instead of copying code. Distinguish verified
facts, inferences and unknowns.

## 6. Leaving the repository

Before you stop: the working tree should build and its tests should behave as documented;
uncommitted work you created should be described in PROJECT_STATE.md (what it is and
whether it is finished); temporary files should be removed; and new output directories
under `outputs/` should be named so the next agent can tell what produced them.
