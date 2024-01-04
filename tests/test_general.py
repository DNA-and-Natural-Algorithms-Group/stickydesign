from stickydesign import energetics_daoe
from stickydesign import energetics_basic
from stickydesign import EndArray
import pytest
import numpy as np
from stickydesign.energetics_basic import ACCEL

@pytest.fixture()
def sets_daoe():
        r5dt = EndArray(np.random.randint(low=0, high=4, size=(100, 7)), 'DT')
        r5td = EndArray(np.random.randint(low=0, high=4, size=(100, 7)), 'TD')
        r10dt = EndArray(
            np.random.randint(low=0, high=4, size=(100, 12)), 'DT')
        r10td = EndArray(
            np.random.randint(low=0, high=4, size=(100, 12)), 'TD')
        return [r5dt, r5td, r10dt, r10td]

@pytest.fixture()
def sets_basic():
    r7s = EndArray(np.random.randint(low=0, high=4, size=(100, 7)), 'S')
    r12s = EndArray(np.random.randint(low=0, high=4, size=(100, 12)), 'S')
    return [r7s, r12s]

@pytest.fixture()
def en_daoe():
    return energetics_daoe.EnergeticsDAOE()

@pytest.fixture()
@pytest.mark.skipif(not ACCEL, reason="No acceleration available.")
def en_basicold():
    from stickydesign import energetics_basic_old
    return energetics_basic_old.EnergeticsBasicOld()

@pytest.fixture()
def en_basic():
    return energetics_basic.EnergeticsBasic()


def test_daoe_matching_energies_match(sets_daoe, en_daoe):
    for s in sets_daoe:
        r1 = en_daoe.matching_uniform(s)
        r2 = en_daoe.uniform(s.ends, s.comps)
        print(repr(s))
        np.testing.assert_array_almost_equal(r1, r2)

def test_daoe_symmetry(sets_daoe, en_daoe):
    for s in sets_daoe:
        r1 = en_daoe.uniform(s[:50], s[50:])
        r2 = en_daoe.uniform(s[50:], s[:50])
        np.testing.assert_array_almost_equal(r1, r2)

def test_basic_matching_energies_match(sets_basic, en_basic):
    for s in sets_basic:
        r1 = en_basic.matching_uniform(s)
        r2 = en_basic.uniform(s.ends, s.comps)
        print(repr(s))
        np.testing.assert_array_almost_equal(r1, r2)

def test_basic_symmetry(sets_basic, en_basic):
    for s in sets_basic:
        r1 = en_basic.uniform(s[:50], s[50:])
        r2 = en_basic.uniform(s[50:], s[:50])
        np.testing.assert_array_almost_equal(r1, r2)

@pytest.mark.skipif(not ACCEL, reason="No acceleration available.")
def test_basic_old_matching_energies_match(sets_daoe, en_basicold):
    for s in sets_daoe:
        r1 = en_basicold.matching_uniform(s)
        r2 = en_basicold.uniform_loopmismatch(s.ends, s.comps)
        r3 = en_basicold.uniform_danglemismatch(s.ends, s.comps)
        print(repr(s))
        np.testing.assert_array_almost_equal(r1, r2)
        np.testing.assert_array_almost_equal(r1, r3)

@pytest.mark.skipif(not ACCEL, reason="No acceleration available.")
def test_basic_old_symmetry(sets_daoe, en_basicold):
    for s in sets_daoe:
        r1 = en_basicold.uniform(s[:50], s[50:])
        r2 = en_basicold.uniform(s[50:], s[:50])
        np.testing.assert_array_almost_equal(r1, r2)
