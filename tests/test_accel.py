import pytest
import stickydesign as sd
import numpy as np
import random
from typing import cast
from numpy.testing import assert_array_equal
#import hypothesis

@pytest.fixture
def random_seqs_with_complements() -> sd.EndArray:
    L = 10
    N = 50
    e = sd.EndArray(["".join(random.choices("acgt", k=L)) for _ in range(N)], 'S')
    e = e.append(e.comps)
    return e

@pytest.fixture
def seqarr(random_seqs_with_complements) -> tuple[sd.EndArray, sd.EndArray]:
    rsc = random_seqs_with_complements
    a = cast(sd.EndArray, np.repeat(rsc, rsc.shape[0], 0))
    b = cast(sd.EndArray, np.tile(rsc, (rsc.shape[0], 1)))
    return a, b

@pytest.fixture
def slowres(seqarr):
    en_slow = sd.EnergeticsBasic(_tightloop=False, _accel=False)
    return en_slow.uniform(*seqarr)

def test_nonaccel(seqarr, slowres, benchmark):
    en = sd.EnergeticsBasic(_tightloop=False, _accel=False)
    
    res = benchmark(en.uniform, *seqarr)
    
    assert_array_equal(res, slowres)

@pytest.mark.skipif(not sd.energetics_basic.ACCEL, reason="No acceleration available.")
def test_tightloop(seqarr, slowres, benchmark):
    en_fast = sd.EnergeticsBasic(_tightloop=True, _accel=False)

    b = benchmark(en_fast.uniform, *seqarr)

    assert_array_equal(slowres, b)

@pytest.mark.skipif(not sd.energetics_basic.ACCEL, reason="No acceleration available.")
def test_accel(seqarr, slowres, benchmark):
    en_fast = sd.EnergeticsBasic(_accel=True)

    b = benchmark(en_fast.uniform, *seqarr)

    assert_array_equal(slowres, b)