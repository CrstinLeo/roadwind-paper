"""Conventions of the air-drag module: wind 'from' direction, head/cross components, VECTO yaw curve."""
import numpy as np

from smhi_trv_wind import drag


def test_components_convention():
    # vehicle heading north (0 deg); wind FROM north is a pure headwind, FROM east a pure crosswind
    h, c = drag.components(5.0, 0.0, 0.0)
    assert np.isclose(h, 5.0) and np.isclose(c, 0.0, atol=1e-12)
    h, c = drag.components(5.0, 180.0, 0.0)
    assert np.isclose(h, -5.0)                                    # tailwind
    h, c = drag.components(5.0, 90.0, 0.0)
    assert np.isclose(h, 0.0, atol=1e-12) and np.isclose(abs(c), 5.0)


def test_still_air_energy():
    v = 80 / 3.6
    e0 = drag.energy_kwh100(v, 0.0, 0.0, rho=1.25)
    assert np.isclose(e0, 0.5 * 1.25 * drag.CDA0 * v ** 2 / 36)
    assert np.isclose(e0, 49.7, atol=0.1)


def test_vecto_curve():
    b = np.radians([0.0, 10.0, drag.BETA_PEAK, 20.0])
    r = drag.cda(b, "vecto") / drag.CDA0
    assert np.isclose(r[0], 1.0)
    assert np.isclose(r[1], 1.39, atol=0.01)                       # +39 % at 10 deg
    assert np.isclose(drag.BETA_PEAK, 13.13, atol=0.01)
    assert np.isclose(r[3], r[2])                                   # held constant beyond the peak
    assert np.all(np.diff(drag.cda(np.radians(np.linspace(0, drag.BETA_PEAK, 50)), "vecto")) >= 0)


def test_mirror_invariance():
    # reflecting the wind about the vehicle axis (cross -> -cross) leaves the drag unchanged
    v = 80 / 3.6
    assert np.isclose(drag.energy_kwh100(v, 2.0, 3.0), drag.energy_kwh100(v, 2.0, -3.0))


def test_round_trip_penalty_positive():
    # with the yaw-dependent drag any wind raises the round-trip mean above still air
    v = 80 / 3.6
    e0 = drag.energy_kwh100(v, 0.0, 0.0)
    for wfrom in (0.0, 45.0, 90.0):
        e = [drag.energy_kwh100(v, *drag.components(6.0, wfrom, hd)) for hd in (0.0, 180.0)]
        assert np.mean(e) > e0
