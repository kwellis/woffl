"""Batchpump Smoke Tests

Smoke tests for batch_run, process_results, and search_run.
Uses the Schrader preset well with a small jet pump grid to keep runtime short.
Asserts physically reasonable outputs — not exact values.
"""

import pytest

from woffl.geometry.jetpump import JetPump
from tests.asm_helper import make_well, jp_list


# ---- fixture ----

@pytest.fixture(scope="module")
def e41_batch():
    """Single well with batch_run and process_results completed."""
    well = make_well("MPE-41", qwf=246, pwf=1049, pres=1400, wc=0.894, fgor=600)
    well.batch_run(jp_list)
    well.process_results()
    return well


# ---- batch_run tests ----

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


# ---- process_results tests ----

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


# ---- search_run tests ----

class TestSearchRun:
    """Smoke tests for BatchPump.search_run"""

    def test_returns_catalog_pump(self):
        well = make_well("MPE-41", qwf=246, pwf=1049, pres=1400, wc=0.894, fgor=600)
        seed = JetPump("12", "B")
        df = well.search_run(seed, lift_cost=0.03)
        assert len(df) == 1
        assert df["nozzle"].iloc[0] != "opt"
        assert df["throat"].iloc[0] != "opt"

    def test_oil_rate_positive(self):
        well = make_well("MPE-41", qwf=246, pwf=1049, pres=1400, wc=0.894, fgor=600)
        seed = JetPump("12", "B")
        df = well.search_run(seed, lift_cost=0.03)
        assert df["qoil_std"].iloc[0] > 0

    def test_does_not_overwrite_batch_df(self):
        well = make_well("MPE-41", qwf=246, pwf=1049, pres=1400, wc=0.894, fgor=600)
        well.batch_run(jp_list)
        batch_len = len(well.df)
        seed = JetPump("12", "B")
        well.search_run(seed, lift_cost=0.03)
        assert len(well.df) == batch_len, "search_run overwrote self.df"
