"""No-input resting-state full-CNS control (100 ms by default)."""

from malecns_sim.neural.experiments.common import main

if __name__ == "__main__":
    main("full_cns", spontaneous=True)
