import math

import pytest

from woffl.geometry.jetpump import JetPump


class TestJetPumpInit:
    """Test that JetPump correctly looks up catalog nozzle and throat sizes."""

    def test_known_pump_12c(self):
        """12C is a common pump — verify nozzle and throat from catalog."""
        jp = JetPump("12", "C")
        assert jp.dnz == 0.2099  # nozzle_dia[11]
        assert jp.dth == 0.4335  # throat_dia[13]
        assert jp.noz_no == "12"
        assert jp.rat_ar == "C"

    def test_known_pump_9e(self):
        """9E — small nozzle, large throat ratio."""
        jp = JetPump("9", "E")
        assert jp.dnz == 0.1458  # nozzle_dia[8]
        assert jp.dth == 0.3841  # throat_dia[12]

    def test_known_pump_5a(self):
        """5A — nozzle_idx=4, throat_idx=4+0=4."""
        jp = JetPump("5", "A")
        assert jp.dnz == 0.0903  # nozzle_dia[4]
        assert jp.dth == 0.1458  # throat_dia[4]

    def test_known_pump_7x(self):
        """7X — area_code X is -1, so throat_idx = 6 + (-1) = 5."""
        jp = JetPump("7", "X")
        assert jp.dnz == 0.1145  # nozzle_dia[6]
        assert jp.dth == 0.1643  # throat_dia[5]

    def test_default_loss_coefficients(self):
        jp = JetPump("10", "B")
        assert jp.knz == 0.01
        assert jp.ken == 0.03
        assert jp.kth == 0.3
        assert jp.kdi == 0.3

    def test_custom_loss_coefficients(self):
        jp = JetPump("10", "B", knz=0.02, ken=0.05, kth=0.4, kdi=0.5)
        assert jp.knz == 0.02
        assert jp.ken == 0.05
        assert jp.kth == 0.4
        assert jp.kdi == 0.5


class TestJetPumpErrors:
    """Test that bad nozzle/throat combos raise ValueError."""

    def test_nozzle_too_large(self):
        with pytest.raises(ValueError, match="Nozzle size"):
            JetPump("25", "A")

    def test_nozzle_zero(self):
        with pytest.raises(ValueError, match="out of range"):
            JetPump("0", "A")

    def test_nozzle_negative(self):
        with pytest.raises(ValueError, match="out of range"):
            JetPump("-5", "A")

    def test_nozzle_non_numeric(self):
        with pytest.raises(ValueError):
            JetPump("abc", "A")

    def test_invalid_area_ratio(self):
        with pytest.raises(ValueError, match="not recognized"):
            JetPump("10", "Z")

    def test_throat_index_out_of_range(self):
        """20E — nozzle exists (index 19) but throat_idx = 19+4 = 23, out of range."""
        with pytest.raises(ValueError, match="Nozzle throat combo"):
            JetPump("20", "E")


class TestJetPumpProperties:
    """Test computed areas from catalog diameters."""

    def test_nozzle_area(self):
        jp = JetPump("12", "C")
        expected = math.pi * (0.2099**2) / 4 / 144
        assert jp.anz == pytest.approx(expected)

    def test_throat_area(self):
        jp = JetPump("12", "C")
        expected = math.pi * (0.4335**2) / 4 / 144
        assert jp.ath == pytest.approx(expected)

    def test_throat_entry_area(self):
        """ate = ath - anz, the annular area between nozzle OD and throat ID."""
        jp = JetPump("12", "C")
        assert jp.ate == pytest.approx(jp.ath - jp.anz)

    def test_throat_larger_than_nozzle(self):
        """For any valid pump, throat area must exceed nozzle area."""
        jp = JetPump("10", "A")
        assert jp.ath > jp.anz
        assert jp.ate > 0
