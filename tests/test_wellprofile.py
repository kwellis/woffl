import numpy as np
import pytest

from woffl.geometry.wellprofile import WellProfile


sch_profile = WellProfile.schrader()
# sch_profile.plot_raw()
# sch_profile.plot_filter()

kup_profile = WellProfile.kuparuk()
# kup_profile.plot_raw()
# kup_profile.plot_filter()


def test_sch_vs_kup() -> None:
    assert sch_profile.vd_ray[-1] == pytest.approx(4500, 0.1)
    assert kup_profile.vd_ray[-1] == pytest.approx(9000, 0.1)


# Sample Data
md_list = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
vd_list = [0, 95, 180, 270, 350, 375, 400, 410, 415, 420, 425]
jetpump_md = 450
wp = WellProfile(md_list, vd_list, jetpump_md)


def test_initialization():
    """Test WellProfile initialization with valid inputs"""
    assert len(wp.md_ray) == len(wp.vd_ray) == len(wp.md_ray)
    assert len(wp.md_fit) == len(wp.vd_fit) == len(wp.md_fit)
    assert wp.jetpump_md == jetpump_md


def test_initialization_invalid_lengths():
    """Test initialization failure for mismatched lengths"""
    with pytest.raises(ValueError, match="Lists for Measured Depth and Vertical Depth need to be the same length"):
        WellProfile([0, 100], [0, 80, 150], 100)


def test_initialization_invalid_order():
    """Test initialization failure for measured depth being too short"""
    with pytest.raises(ValueError, match="Measured Depth needs to extend farther than Vertical Depth"):
        WellProfile([0, 80, 150], [0, 100, 200], 100)


def test_init_invalid_jetpump() -> None:
    with pytest.raises(ValueError, match="Jet pump not inside well profile measured depth"):
        WellProfile(md_list, vd_list, jetpump_md=1200)


def test_vd_interp():
    """Test vertical depth interpolation"""
    item_md = 350
    assert wp.vd_interp(item_md) <= item_md
    assert wp.vd_interp(item_md) == pytest.approx(310, 0.05)


def test_hd_interp():
    """Test horizontal distance interpolation"""
    item_md = 350
    assert wp.hd_interp(item_md) <= item_md
    assert wp.hd_interp(item_md) == pytest.approx(160, 0.05)


def test_md_interp():
    """Test measured depth interpolation"""
    item_vd = 190
    assert wp.md_interp(item_vd) >= item_vd
    assert wp.md_interp(item_vd) == pytest.approx(210, 0.01)


def test_filter():
    """Test profile filtering reduces number of points while maintaining key structure"""
    assert len(wp.md_fit) < len(wp.md_ray)  # Filtering should reduce points
    assert wp.md_fit[0] == 0  # First point should always be 0
    assert wp.vd_fit[0] == 0  # First point should always be 0


def test_outflow_spacing():
    """Test outflow spacing generates valid segments"""
    md_seg, vd_seg = wp.outflow_spacing(50)  # 50-ft segments
    assert len(md_seg) == len(vd_seg)  # Should be same length
    assert md_seg[-1] == jetpump_md  # outflow ends at jetpump depth
    assert md_seg[-1] <= max(md_list)  # Should not exceed max depth
