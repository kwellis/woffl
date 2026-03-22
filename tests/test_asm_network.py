"""Network Smoke Tests

Smoke tests for optimize_jet_pumps (MCKP solver).
Uses the Schrader preset with three wells to keep runtime short.
Asserts physically reasonable outputs — not exact values.
"""

import pytest

from woffl.assembly.network import optimize_jet_pumps
from tests.asm_helper import make_well, jp_list


# ---- fixture ----

well_configs = [
    {"name": "MPE-41", "qwf": 246, "pwf": 1049, "pres": 1400, "wc": 0.894, "fgor": 600},
    {"name": "MPE-42", "qwf": 180, "pwf": 1100, "pres": 1350, "wc": 0.920, "fgor": 550},
    {"name": "MPE-43", "qwf": 310, "pwf": 950, "pres": 1450, "wc": 0.850, "fgor": 650},
]


@pytest.fixture(scope="module")
def network_wells():
    """Three wells with batch_run and process_results completed."""
    wells = []
    for cfg in well_configs:
        well = make_well(cfg["name"], cfg["qwf"], cfg["pwf"], cfg["pres"], cfg["wc"], cfg["fgor"])
        well.batch_run(jp_list)
        well.process_results()
        wells.append(well)
    return wells


# ---- MCKP tests ----

class TestOptimizeJetPumps:
    """Smoke tests for optimize_jet_pumps (MCKP solver)"""

    def test_one_row_per_well(self, network_wells):
        df = optimize_jet_pumps(network_wells, qpf_tot=6000)
        assert len(df) == len(network_wells)

    def test_oil_rates_positive(self, network_wells):
        df = optimize_jet_pumps(network_wells, qpf_tot=6000)
        assert (df["qoil_std"] > 0).all()

    def test_capacity_respected(self, network_wells):
        qpf_tot = 6000
        df = optimize_jet_pumps(network_wells, qpf_tot=qpf_tot)
        assert df["lift_wat"].sum() <= qpf_tot

    def test_tight_capacity_respected(self, network_wells):
        """Set capacity tight enough to bind the constraint."""
        min_water = sum(
            well.df[well.df["semi"]]["lift_wat"].min() for well in network_wells
        )
        qpf_tot = min_water + 100  # just above minimum
        df = optimize_jet_pumps(network_wells, qpf_tot=qpf_tot)
        assert df["lift_wat"].sum() <= qpf_tot

    def test_infeasible_raises(self, network_wells):
        with pytest.raises(RuntimeError, match="infeasible"):
            optimize_jet_pumps(network_wells, qpf_tot=10)
