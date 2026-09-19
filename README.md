# malecns-sim v0.4 — MaleCNS I/O Catalogue

v0.4 adds a versioned anatomical interface catalogue over the v0.2 graph. It
answers which retained MaleCNS cells are sensory, motor, descending, ascending,
intrinsic, endocrine, or other, while preserving every released annotation used
to make that decision. v0.3 still provides the Shiu-derived Brian2 LIF engine.
**The FlyGym/MuJoCo body remains independent:** no encoder, motor decoder,
actuator mapping, sensory feedback, training, or CNS-to-joint connection exists.

## Start with the I/O catalogue

The raw files and `published-v1` graph must already exist. Run these commands
from the repository root:

```powershell
uv sync --locked

# Audit the real released schema and every classification value
uv run python -m malecns_sim.io.audit

# Build once; this refuses to overwrite an existing io-v1 directory
uv run python -m malecns_sim.io.build

# Verify graph identity, raw metadata and all normalized derivations
uv run python -m malecns_sim.io.validate data/processed/malecns-v1.0/published-v1/io-v1

# Print role counts and query exact annotation values
uv run python -m malecns_sim.io.inspect
uv run python -m malecns_sim.io.query --role motor --side L
uv run python -m malecns_sim.io.query --role sensory --class mechanosensory_proprioceptive --subclass leg
uv run python -m malecns_sim.io.query --role descending --type DNa01

# Report configured FlyGym/MuJoCo channels; this does not map or move the body
uv run python -m malecns_sim.simulation.body_io
```

The built catalogue contains 166,700 rows: 17,937 sensory, 815 motor, 1,316
descending, 1,846 ascending, 144,494 interneuron, 94 endocrine, and 198 other.
All graph nodes match an explicit released `superclass` rule. These are policy
counts over the provisional 166,700-node graph; the published 166,691 census
discrepancy described below remains unresolved.

Run the first anatomy-aware neural experiment with a real population:

```powershell
uv run python -m malecns_sim.io.experiment `
    --class mechanosensory_proprioceptive `
    --subclass leg `
    --duration-ms 100 `
    --rate-hz 100 `
    --seed 42 `
    --output outputs/io/leg_proprio_demo
```

This stimulates 190 released leg proprioceptive sensory annotations in the full
LIF graph and writes `active_neurons.parquet` plus role-level activity. It is an
anatomical connectivity experiment, not a FlyGym sensor encoder or a biological
locomotion claim. Output directories are never overwritten.

Read [the I/O Catalogue guide](docs/io-catalogue.md) for exact role rules,
normalization provenance, query APIs, result schemas, body channel inventory,
measured acceptance results, and Docker commands.

## Start with the neural model

Open PowerShell in the repository root. For an existing local installation,
update dependencies first:

```powershell
uv sync --locked
uv run pytest tests/neural -m "not graph" -q
uv run python -m malecns_sim.neural.experiments.toy --seed 42
uv run python -m malecns_sim.neural.visualize outputs/neural/toy --toy-latency
```

The toy experiment needs no downloaded data. It writes spike times, firing
rates, model parameters and a summary to `outputs/neural/toy/`. The commands
below need the built `published-v1` graph; follow the graph setup immediately
below if it is missing.

```powershell
# 5,000-node maximum; 100 ms baseline, 500 ms stimulation, 400 ms recovery
uv run python -m malecns_sim.neural.experiments.subgraph --seed 42 --output outputs/neural/subgraph_demo

# Full graph: start with only 100 ms, stimulating throughout this short run
uv run python -m malecns_sim.neural.experiments.full_cns `
    --duration-ms 100 `
    --stimulate-body-id 11755 `
    --rate-hz 100 `
    --seed 42 `
    --output outputs/neural/full_demo

# Read the results without starting another simulation
Get-Content outputs/neural/full_demo/summary.json
uv run python -m malecns_sim.neural.result outputs/neural/full_demo
```

`11755` is a real retained body ID in the supplied MaleCNS v1.0 graph. It was
chosen by the strongest positive-to-positive non-self connection, **not a
behavioral or sensory annotation**. Output directories are never overwritten;
choose a new `--output` for each experiment.

For Docker, build the updated image, then use the same mounted graph:

```powershell
docker compose build
docker compose run --rm sim python -m malecns_sim.neural.experiments.toy --output outputs/neural/toy_docker
docker compose run --rm sim python -m malecns_sim.neural.experiments.full_cns --duration-ms 100 --stimulate-body-id 11755 --rate-hz 100 --seed 42 --output outputs/neural/full_docker
uv run python -m malecns_sim.neural.result outputs/neural/full_demo --compare outputs/neural/full_docker
```

Read [the Neural Reference guide](docs/neural-reference.md) for the exact model,
unknown-NT policy, subgraph selection, stimulus API, result schema, reproducibility
checks and measured acceptance results. New installation instructions for
Windows/uv and Docker are further down this page.

## Build or inspect the v0.2 graph

For graph setup, policy definitions, storage layout, queries, and acceptance
checks, read [the Graph Core guide](docs/graph-core.md). The Windows/Docker body
setup below is still applicable.

After installing the environment and downloading the raw data, run from the
repository root (choose either local uv or Docker):

```powershell
uv run python -m malecns_sim.cns.census
uv run python -m malecns_sim.cns.graph.build --policy published
uv run python -m malecns_sim.cns.graph.validate data/processed/malecns-v1.0/published-v1
```

```powershell
docker compose build
docker compose run --rm sim python -m malecns_sim.cns.census
docker compose run --rm sim python -m malecns_sim.cns.graph.build --policy published
docker compose run --rm sim python -m malecns_sim.cns.graph.validate data/processed/malecns-v1.0/published-v1
```

Both workflows share the host's `data/` directory. **Build once**, then load or
validate the result. The builder refuses to overwrite an existing directory;
use `--output data/processed/malecns-v1.0/another-run` for a separate build.

The default `published` profile retains **166,700 node candidates**, compared with
the publication's **166,691 neurons**. The **+9 discrepancy remains unresolved**;
this profile is explicitly provisional, not an exact published membership claim.
The actual build contains **25,582,938 unique directed edges** and **124,177,617
contacts**. See the guide for the census audit and exclusion accounting.

Run the graph tests after building:

```powershell
uv run pytest tests/test_graph.py -q
```

For small graph fixtures without the full built graph:

```powershell
uv run pytest tests/test_graph.py -m "not graph" -q
```

The Python 3.12 environment contains **MaleCNS v1.0**, the graph and neural model,
and a **FlyGym 2.1 / MuJoCo 3.9** physical body. The CNS and body are **not connected yet**.
The working directory may be named `flycns-sim`; the Python package is `malecns_sim`.

```text
src/malecns_sim/
├── cns/          Raw data inspection / census / deterministic anatomical graph
├── neural/       Transmitter policy / Brian2 LIF / stimuli / spikes and rates
├── io/           Functional role catalogue / selectors / anatomy-aware reports
└── simulation/   NeuroMechFly → FlatGroundWorld → MjModel / MjData → physics / render
```

The raw `CNSDataset` reader leaves released `minconf-0.5` tables unchanged.
`cns/graph/` applies a documented node policy and preserves anatomical contact
counts. Only `neural/` assigns transmitter signs and model weights.
`io/` derives functional interface annotations without changing the graph.
`simulation/` does not access CNS data. This version makes no claim about walking
or biological behavior.

## Which workflow should I use?


| Your goal                                       | Workflow              | Result                                               |
| ------------------------------------------------- | ----------------------- | ------------------------------------------------------ |
| Verify the setup and generate simulation images | Docker / headless     | Eight PNG files and a JSON report; no window opens   |
| See the fly in a live window                    | Windows / uv / viewer | A MuJoCo application window                          |
| Inspect counts and labels in MaleCNS tables     | CNS inspect           | A data summary in the terminal; no simulation starts |

Use the Docker workflow for your first run. **You do not need to download MaleCNS
data to run the body.** Downloading data is an independent step.
In v0.1, the fly is held in a neutral pose and interacts physically with the
ground. Walking control and movement commands from the CNS are not implemented.

## 1. Open the project directory

Open **PowerShell** from the Windows Start menu. This command opens the project
directory on the original development computer:

```powershell
cd C:\Users\USER\Desktop\DOA\projects\flycns-sim
```

If you copied the project elsewhere, use your own directory path. Check that
you are in the correct directory:

```powershell
Get-Item .\README.md, .\pyproject.toml, .\docker-compose.yml
```

All three files should be listed. Run every project command below **from this
directory**, waiting for each command to finish and the PowerShell prompt to
return before running the next one. Copy the commands inside the code blocks,
without the Markdown fence characters.

## 2. Set up and run with Docker

### 2.1. Prepare Docker Desktop

Git and Docker Desktop were already installed on the original development
computer. On another computer, first install
[Git for Windows](https://git-scm.com/install/windows) and
[Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/)
using their official installers. Select the WSL 2 backend during Docker setup
and restart the computer if the installer requests it.

1. Open **Docker Desktop** from the Start menu.
2. Check **Settings → General → Use the WSL 2 based engine**.
3. Make sure Docker is using Linux containers and the engine is running.
4. Run these checks in PowerShell:

```powershell
docker version
docker compose version
wsl --status
```

The `docker version` output should include both **Client and Server** sections.
The Compose command should print a version. If Docker or WSL setup fails,
complete the requirements in the
[Windows installation guide](https://docs.docker.com/desktop/setup/install/windows-install/).
You do not need to install Python or uv on Windows for this workflow.

### 2.2. Build the project image

Do this during initial setup or after changing Python code or dependencies:

```powershell
docker compose build
```

This installs Python 3.12, FlyGym, MuJoCo, and the other packages into the image.
The first build downloads dependencies and may take a while. Wait for the command
to finish successfully before continuing. **You do not need to rebuild before
every simulation.** Skip this step if you already built the image for this
directory and have not changed the code.

### 2.3. Run the simulation

```powershell
docker compose run --rm sim python -m malecns_sim.simulation.smoke_test --headless
```

The command creates the body and flat ground, runs physics, and exits when done.
`--rm` removes the temporary container after it exits; data and output files on
the host remain. **No separate window is expected in this mode.**

The default run is **2 seconds of simulated time / 20,000 physics steps**.
Computing this may take longer than 2 seconds on your computer. The terminal
should first show `Compiled NeuroMechFly`, followed by a report containing fields
similar to these:

```json
{
  "steps": 20000,
  "simulated_seconds": 1.9999999999997962,
  "finite_state": true,
  "warning_count": 0
}
```

This is an abbreviated report example. A duration very close to `2.0` is normal
because of numerical rounding. `finite_state: true` and `warning_count: 0` are
the expected results.

### 2.4. Open the images

Once the command finishes:

```powershell
explorer .\outputs\smoke_test
```

Double-click the eight images, `frame_000.png` through `frame_007.png`, in the
folder that opens. `report.json` contains the step count, duration, and validation
results. To read the report in PowerShell:

```powershell
Get-Content .\outputs\smoke_test\report.json
```

This version does not produce MP4 files. Running the same command again replaces
the PNGs and report with the same names. See the example below for saving
different runs separately.

### 2.5. Run again later

Open Docker Desktop, wait for the engine to be ready, then run in PowerShell:

```powershell
cd C:\Users\USER\Desktop\DOA\projects\flycns-sim
docker compose run --rm sim python -m malecns_sim.simulation.smoke_test --headless
explorer .\outputs\smoke_test
```

You do not need to repeat installation or data downloads.

## 3. Open a live MuJoCo window on Windows

This workflow requires **Git and uv**. Docker Desktop does not need to be running.
The Windows `.venv` environment is separate from the Docker image, so its packages
are installed separately. Launching the Windows viewer has not yet been manually
verified for this project; the Docker headless workflow passed the previous
acceptance run.

### 3.1. Install uv

Run this command in PowerShell:

```powershell
winget install --id=astral-sh.uv -e
```

This is one of [uv's official Windows installation options](https://docs.astral.sh/uv/getting-started/installation/).
If `winget` is unavailable, use the Windows standalone installer described on the
same page. If uv is already installed, you do not need to install it again.

Close and reopen PowerShell after installation. Check:

```powershell
uv --version
git --version
```

Both commands should print a version number.

### 3.2. Prepare the local Python environment

```powershell
cd C:\Users\USER\Desktop\DOA\projects\flycns-sim
uv sync --locked
```

uv downloads Python 3.12 if needed, creates `.venv` inside the project, and installs
the dependencies from `uv.lock`. The project uses Python 3.12 even if another
Python version is installed on Windows. You do not need to activate `.venv`
manually; run commands with `uv run`. Initial setup requires an internet connection.

### 3.3. Open the viewer and start physics

```powershell
uv run python -m malecns_sim.simulation.smoke_test --viewer
```

A MuJoCo window opens. Press **Space** to start or pause physics playback.
Close the window when finished. The terminal remains busy while the window is
open. Viewer mode does not save PNGs or JSON reports; use the headless command
for file output.

For subsequent launches, run just the viewer command from the project directory.
Do not add Docker's `MUJOCO_GL=osmesa` and `PYOPENGL_PLATFORM=osmesa` settings to
your Windows environment.

## 4. Download and inspect MaleCNS data

These steps prepare the CNS data layer; they do not move the fly. **The three
files were downloaded on the original development computer during initial repo
setup.** If they are still present, go straight to the inspection command.
On a new computer, or if files are missing, download them first:

```powershell
docker compose run --rm sim python -m malecns_sim.cns.download
```

The total download is approximately 1.1 GB. When the command finishes, the three
files are in `data\malecns-v1.0\`. Existing valid files are checked for size and
checksum instead of being downloaded again. If a download is interrupted, rerun
the same command: the incomplete file is downloaded from the beginning, while
completed files are preserved.

Then inspect the data:

```powershell
docker compose run --rm sim python -m malecns_sim.cns.inspect
```

The terminal prints annotation, neurotransmitter, and connection counts;
columns; and status and neurotransmitter labels. There may be a delay before
the first output while the tables load into RAM. The previous full dataset
check reported these row counts:


| Table             |        Rows |
| ------------------- | ------------: |
| Annotations       |     211,577 |
| Neurotransmitters |   1,835,518 |
| Connections       | 151,856,684 |

If you have set up the local uv environment, you can do the same without Docker:

```powershell
uv run python -m malecns_sim.cns.download
uv run python -m malecns_sim.cns.inspect
```

When run from the repository root, both workflows use the same `data/` directory.
You do not need to download the data twice.

## Raw data and Docker file layout

The Docker image includes Python packages, test tools, and OSMesa; CNS data is
stored separately. `data/` and `outputs/` are mounted as host volumes, and
`.dockerignore` excludes them from the build context. Inside the container,
`/app/data` maps to the project's `data/` directory on your computer, and
`/app/outputs` maps to its `outputs/` directory. Docker's OpenGL settings are
already configured in the Compose file; no GPU or GUI forwarding is required.

Only these three files are downloaded into `data/malecns-v1.0/`:


| Table             | File                                                   | Approximate download |
| ------------------- | -------------------------------------------------------- | ---------------------- |
| Annotations       | `body-annotations-male-cns-v1.0-minconf-0.5.feather`   | 13 MB                |
| Neurotransmitters | `body-neurotransmitters-male-cns-v1.0.feather`         | 42 MB                |
| Connections       | `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1.1 GB               |

The downloader retrieves the GCS object size and MD5 over HTTPS, downloads that
same object generation to a `.part` file, then moves it to its final name after
checksum verification. Existing valid files are reused. Rerun the command after
an interruption; the affected file starts again from the beginning, and other
completed files are preserved. Checksums verify transfer integrity, not
scientific validity.

`CNSDataset.load()` reads all three tables into RAM with `pandas.read_feather()`.
Memory use can exceed the compressed size on disk. Allocate enough memory to
Docker for full dataset tests. `inspect` reports row counts, columns, dtypes, and
status and neurotransmitter labels, including missing values. Missing or corrupt
files produce an error. Both CNS commands accept `--data-dir PATH` to use a
different directory.

Data and render outputs are excluded from Git. Skeletons, syn-points,
syn-partners, and EM volumes are not downloaded. MaleCNS data is subject to the
publisher's CC-BY license; follow the official attribution and licensing terms
when using it for research.

## 5. Test the setup

After downloading the three CNS files, run the full test suite with Docker:

```powershell
docker compose run --rm sim pytest tests/test_cns.py tests/test_simulation.py -q
```

The previous v0.1 acceptance run reported **14 passed**. Runtime depends on your
computer. To run the same test suite in the local uv environment:

```powershell
uv run pytest tests/test_cns.py tests/test_simulation.py -q
```

`pytest` runs the four real acceptance checks below, plus small tests for failure
paths. Missing data is not silently skipped: dataset tests fail and show the
download command. Tests do not automatically download large files.

1. All three official MaleCNS files exist and are nonempty.
2. All three real Feather files open, including the connection table.
3. NeuroMechFly and FlatGroundWorld compile into real `MjModel`/`MjData` objects.
4. Exactly 1,000 physics steps complete with finite state, advancing simulation
   time, zero MuJoCo warnings, and readable, nonblank headless PNGs.

To check just the code and body without downloading data, choose either workflow:

```powershell
docker compose run --rm sim pytest tests/test_cns.py tests/test_simulation.py -q -m "not dataset"
```

```powershell
uv run pytest tests/test_cns.py tests/test_simulation.py -q -m "not dataset"
```

This selection excludes dataset tests and does not replace the full v0.1
acceptance check. Test PNGs are written to temporary test directories; use the
headless simulation command to generate persistent images you can inspect.

After editing code, check local lint and formatting:

```powershell
uv run ruff check .
uv run ruff format --check .
```

## 6. Change the duration and output directory

The default headless command runs **2 seconds of simulated time** and writes
eight PNGs plus `report.json` to `outputs/smoke_test/`. Simulated time differs from
wall-clock time. Output is a PNG sequence, so MP4 encoding is not required.

```powershell
docker compose run --rm sim python -m malecns_sim.simulation.smoke_test --headless --steps 1000
```

This runs exactly 1,000 physics steps: **0.1 seconds of simulated time** at the
current timestep. To specify two seconds explicitly:

```powershell
docker compose run --rm sim python -m malecns_sim.simulation.smoke_test --headless --duration 2
```

To preserve earlier outputs and write to a new directory:

```powershell
docker compose run --rm sim python -m malecns_sim.simulation.smoke_test --headless --duration 2 --output-dir outputs/deneme-01
explorer .\outputs\deneme-01
```

`--steps` and `--duration` cannot be used together. In Docker, keep
`--output-dir` **under outputs/** so the files persist on the host. The example
path is `/app/outputs/deneme-01` inside the container and `outputs\deneme-01`
inside the project on Windows. Subsequent runs replace files with the same names
in the selected directory.

## 7. Troubleshooting


| Situation                                                      | What to do                                                                                                                                                                   |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| The`docker` or `uv` command is not found                       | Install the tool, close and reopen PowerShell, then retry its version command.                                                                                               |
| Docker reports`Cannot connect` or `error during connect`       | Open Docker Desktop, wait for the engine, and check that`docker version` includes a Server section.                                                                          |
| `no configuration file provided`                               | You are in the wrong directory. Use`cd` to enter the repository root containing `docker-compose.yml`.                                                                        |
| The Docker image is missing or a build fails                   | Run`docker compose build` from the project root. Retry the simulation once the build succeeds.                                                                               |
| The simulation finished but no window opened                   | `--headless` produces PNGs. Open `outputs\smoke_test`. Follow the Windows/uv section for a live window.                                                                      |
| `Missing MaleCNS files`                                        | Run the CNS download command. Data is not required to run just the body.                                                                                                     |
| A CNS download was interrupted or failed checksum verification | Check your connection and rerun the download command; valid files are preserved.                                                                                             |
| Memory runs out during`inspect` or the full test suite         | Loading the full table requires several GB of RAM plus space for temporary copies. Close other memory-heavy applications and check the WSL 2 memory limit.                   |
| The Windows viewer reports an OpenGL/OSMesa error              | Try the session cleanup below and reopen the viewer. If it persists, check your graphics driver/OpenGL environment; the verified Docker headless workflow is also available. |
| Code changes do not appear in Docker                           | Source code is copied into the image. Rebuild with`docker compose build`.                                                                                                    |
| Images look very similar and the fly does not walk             | v0.1 checks the physics of a neutral body; walking control is not implemented yet.                                                                                           |

If OSMesa variables were accidentally set in your Windows PowerShell session,
clear them before retrying the viewer:

```powershell
Remove-Item Env:MUJOCO_GL -ErrorAction SilentlyContinue
Remove-Item Env:PYOPENGL_PLATFORM -ErrorAction SilentlyContinue
uv run python -m malecns_sim.simulation.smoke_test --viewer
```

This removes only the two environment variables from the current terminal
session; it does not delete files. Docker Compose settings remain unchanged.

## Version boundary

v0.1 supplies raw data access and the independent physical body. v0.2 supplies
node policies, census auditing, and sparse COO/CSR connectivity. v0.3 adds the
Shiu-derived Brian2 LIF engine, transmitter policy, scheduled stimuli, and spike
results. Motor mapping, sensory coupling, closed-loop simulation, RL, GPU
backends, GNN dynamics, and Three.js remain outside this release. v0.4 adds the
I/O catalogue, exact queries, anatomy-aware spike reports, and a body channel
inventory; it still defines no mapping across the CNS/body boundary.

## References

- [Official MaleCNS data distribution](https://male-cns.janelia.org/download/)
- [MaleCNS neuPrint query example](https://male-cns.janelia.org/download/)
- [Shiu et al. whole-brain model](https://pmc.ncbi.nlm.nih.gov/articles/PMC11446845/)
- [Shiu reference implementation](https://github.com/philshiu/Drosophila_brain_model/blob/main/model.py)
- [FlyGym v2.1.0 dependencies](https://github.com/NeLy-EPFL/flygym/blob/v2.1.0/pyproject.toml)
- [FlyGym basic model composition](https://neuromechfly.org/tutorials/1a_basic_model_composition/)
- [FlyGym body composition API](https://neuromechfly.org/api_reference/flygym/compose/fly/base_fly/)
- [FlyGym rendering and interactive viewer](https://neuromechfly.org/api_reference/flygym/rendering/)
