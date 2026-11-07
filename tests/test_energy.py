import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import energy as E  # noqa: E402


def test_method_a_hand_calc():
    # 1000 chips x 400 W x 24 h = 9,600 kWh; x load .8 x overhead 1.5 x PUE 1.1 = 12,672 kWh
    assert E.energy_method_a(1000, 400, 24) == pytest.approx(9600)
    assert E.energy_method_a(1000, 400, 24, 0.8, 1.5, 1.1) == pytest.approx(12672)


def test_chip_hours_units():
    # 3.6e18 FLOP at 1e12 FLOP/s, MFU 1 -> 3.6e6 s = 1000 chip-hours
    assert E.chip_hours_from_compute(3.6e18, 1e12, 1.0) == pytest.approx(1000)
    assert E.chip_hours_from_compute(3.6e18, 1e12, 0.5) == pytest.approx(2000)


def test_method_b_consistent_with_a():
    flop, peak, mfu, chips, tdp = 1e23, 312e12, 0.4, 1024, 400
    hours = E.chip_hours_from_compute(flop, peak, mfu) / chips
    assert E.energy_method_b(flop, peak, mfu, tdp) == pytest.approx(E.energy_method_a(chips, tdp, hours))
    assert E.implied_mfu(flop, chips, hours, peak) == pytest.approx(mfu)


def test_order_of_magnitude_large_run():
    # 3e23 FLOP on 125 TFLOP/s, 300 W chips at MFU 0.2 -> ~3.3e6 chip-h -> ~1e6 kWh IT energy
    kwh = E.energy_method_b(3e23, 125e12, 0.2, 300)
    assert kwh == pytest.approx(3e23 / (125e12 * 0.2 * 3600) * 0.3)
    assert 3e5 < kwh < 3e6


def test_co2_units():
    assert E.co2_tonnes(1_000_000, 400) == pytest.approx(400)  # 1 GWh at 400 g/kWh = 400 t


def test_doubling_time():
    assert E.doubling_time_years(np.log10(2)) == pytest.approx(1.0)
    assert E.doubling_time_years(np.log10(4)) == pytest.approx(0.5)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        E.chip_hours_from_compute(1e20, 0, 0.4)


def test_monte_carlo_shapes_bounds_and_seed():
    df = pd.DataFrame({
        "method": ["A", "B"], "tdp_w": [400, 700], "consumer": [False, True],
        "ci_g_kwh": [400, 400], "chips": [100, np.nan], "hours": [10, np.nan],
        "flop": [np.nan, 1e22], "peak_flops": [np.nan, 989e12], "mfu_reported": [np.nan, 0.4],
    })
    e, c, ch, d = E.monte_carlo(df, n=2000, seed=0)
    assert e.shape == (2, 2000) and np.isfinite(e).all() and np.isfinite(c).all()
    lo = 1000 * 0.4 * 0.6 * 1.3 * 1.08   # row A bounds from assumption ranges
    hi = 1000 * 0.4 * 1.0 * 1.9 * 1.60
    assert (e[0] >= lo - 1e-9).all() and (e[0] <= hi + 1e-9).all()
    assert (d["overhead"][1] <= 1.5).all()                     # consumer overhead range used
    assert np.allclose(ch[1], 1e22 / (989e12 * 0.4 * 3600))    # reported MFU used exactly
    e2, *_ = E.monte_carlo(df, n=2000, seed=0)
    assert np.array_equal(e, e2)


def test_tpu_rows_use_measured_system_factor():
    df = pd.DataFrame({"method": ["A", "A"], "tdp_w": [450, 450], "consumer": [False, False], "tpu": [True, False],
                       "ci_g_kwh": [400, 400], "chips": [100, 100], "hours": [10, 10]})
    a = dict(E.ASSUMPTIONS, tpu_system=("const", 0.7))
    e, _, _, d = E.monte_carlo(df, n=1000, seed=1, assumptions=a)
    assert np.allclose(d["overhead"][0], 1.0) and np.allclose(d["load"][0], 0.7)   # TPU: factor only
    assert (d["overhead"][1] >= 1.3).all()                                            # GPU path unchanged
    assert np.allclose(e[0], 1000 * 0.45 * 0.7 * d["pue"][0])
