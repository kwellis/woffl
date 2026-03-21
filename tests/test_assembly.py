"""Assembly Smoke Tests

Smoke tests for batch_run, process_results, search_run, and optimize_jet_pumps.
Uses the Schrader preset well with a small jet pump grid to keep runtime short.
Asserts physically reasonable outputs — not exact values.
"""

import pytest

from woffl.assembly.batchrun import BatchPump
from woffl.assembly.network import optimize_jet_pumps
from woffl.flow.inflow import InFlow
from woffl.geometry.jetpump import JetPump
from woffl.geometry.pipe import Pipe, PipeInPipe
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt.blackoil import BlackOil
from woffl.pvt.formgas import FormGas
from woffl.pvt.formwat import FormWater
from woffl.pvt.resmix import ResMix

# --- shared well setup (MPU E-41 reference) ---

pwh = 210  # psig, wellhead pressure
ppf_surf = 3168  # psig, power fluid surface pressure
tsu = 80  # deg F, suction temperature

tubing = Pipe(out_dia=4.5, thick=0.5)
casing = Pipe(out_dia=6.875, thick=0.5)
wbore = PipeInPipe(inn_pipe=tubing, out_pipe=casing)
profile = WellProfile.schrader()

mpu_oil = BlackOil.schrader()
mpu_wat = FormWater.schrader()
mpu_gas = FormGas.schrader()

# small grid to keep tests fast
nozzles = ["10", "11", "12", "13"]
throats = ["A", "B", "C", "D"]
jp_list = BatchPump.jetpump_list(nozzles, throats)


def _make_well(name, qwf, pwf, pres, wc, fgor):
    """Helper to build a BatchPump from well parameters."""
    ipr = InFlow(qwf=qwf, pwf=pwf, pres=pres)
    res = ResMix(wc=wc, fgor=fgor, oil=mpu_oil, wat=mpu_wat, gas=mpu_gas)
    return BatchPump(
        pwh, tsu, ppf_surf, wbore, profile, ipr, res, mpu_wat,
        jpump_direction="reverse", wellname=name,
    )


# ---- single-well fixtures ----

@pytest.fixture(scope="module")
def e41_batch():
    """Single well with batch_run and process_results completed."""
    well = _make_well("MPE-41", qwf=246, pwf=1049, pres=1400, wc=0.894, fgor=600)
    well.batch_run(jp_list)
    well.process_results()
    return well


# ---- multi-well fixture for MCKP tests ----

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
        well = _make_well(cfg["name"], cfg["qwf"], cfg["pwf"], cfg["pres"], cfg["wc"], cfg["fgor"])
        well.batch_run(jp_list)
        well.process_results()
        wells.append(well)
    return wells


# =====================================================================
# Single-well tests: batch_run, process_results, search_run
# =====================================================================


class TestBatchRun:
    """Smoke tests for BatchPump.batch_run"""

    def test_returns_results(self, e41_batch):
        df = e41_batch.df
        assert len(df) == len(jp_list)

    def test_some_pumps_converge(self, e41_batch):
        df = e41_batch.df
        valid = df.dropna(subset=["qoil_std"])
        assert len(valid) > 0, "No pumps converged"

    def test_oil_rate_positive(self, e41_batch):
        df = e41_batch.df.dropna(subset=["qoil_std"])
        assert (df["qoil_std"] > 0).all()

    def test_suction_pressure_positive(self, e41_batch):
        df = e41_batch.df.dropna(subset=["psu_solv"])
        assert (df["psu_solv"] > 0).all()

    def test_water_rates_positive(self, e41_batch):
        df = e41_batch.df.dropna(subset=["totl_wat"])
        assert (df["totl_wat"] > 0).all()
        assert (df["lift_wat"] > 0).all()


class TestProcessResults:
    """Smoke tests for BatchPump.process_results"""

    def test_semi_finalists_exist(self, e41_batch):
        assert e41_batch.df["semi"].sum() > 0

    def test_gradients_finite(self, e41_batch):
        semi = e41_batch.df[e41_batch.df["semi"]]
        assert semi["motwr"].notna().all()
        assert semi["molwr"].notna().all()

    def test_curve_fit_coefficients(self, e41_batch):
        assert len(e41_batch.coeff_totl) == 3
        assert len(e41_batch.coeff_lift) == 3


class TestSearchRun:
    """Smoke tests for BatchPump.search_run"""

    def test_returns_catalog_pump(self):
        well = _make_well("MPE-41", qwf=246, pwf=1049, pres=1400, wc=0.894, fgor=600)
        seed = JetPump("12", "B")
        df = well.search_run(seed, lift_cost=0.03)
        assert len(df) == 1
        assert df["nozzle"].iloc[0] != "opt"
        assert df["throat"].iloc[0] != "opt"

    def test_oil_rate_positive(self):
        well = _make_well("MPE-41", qwf=246, pwf=1049, pres=1400, wc=0.894, fgor=600)
        seed = JetPump("12", "B")
        df = well.search_run(seed, lift_cost=0.03)
        assert df["qoil_std"].iloc[0] > 0

    def test_does_not_overwrite_batch_df(self):
        well = _make_well("MPE-41", qwf=246, pwf=1049, pres=1400, wc=0.894, fgor=600)
        well.batch_run(jp_list)
        batch_len = len(well.df)
        seed = JetPump("12", "B")
        well.search_run(seed, lift_cost=0.03)
        assert len(well.df) == batch_len, "search_run overwrote self.df"


# =====================================================================
# Multi-well tests: optimize_jet_pumps (MCKP)
# =====================================================================


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
        # find the minimum feasible: smallest pump per well
        min_water = sum(
            well.df[well.df["semi"]]["lift_wat"].min() for well in network_wells
        )
        qpf_tot = min_water + 100  # just above minimum
        df = optimize_jet_pumps(network_wells, qpf_tot=qpf_tot)
        assert df["lift_wat"].sum() <= qpf_tot

    def test_infeasible_raises(self, network_wells):
        with pytest.raises(RuntimeError, match="infeasible"):
            optimize_jet_pumps(network_wells, qpf_tot=10)
