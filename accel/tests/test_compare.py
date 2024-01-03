import pytest
import stickydesign as sd
import numpy as np
import random
from typing import cast
from numpy.testing import assert_array_equal
#import hypothesis

@pytest.fixture
def random_seqs_with_complements() -> sd.endarray:
    L = 10
    N = 100
    e = sd.endarray(["".join(random.choices("acgt", k=L)) for _ in range(N)], 'S')
    e = e.append(e.comps)
    return e

@pytest.fixture
def seqarr(random_seqs_with_complements) -> tuple[sd.endarray, sd.endarray]:
    rsc = random_seqs_with_complements
    a = cast(sd.endarray, np.repeat(rsc, rsc.shape[0], 0))
    b = cast(sd.endarray, np.tile(rsc, (rsc.shape[0], 1)))
    return a, b

def test_tightloop(seqarr):
    en_slow = sd.EnergeticsBasic(_tightloop=False, _accel=False)
    en_fast = sd.EnergeticsBasic(_tightloop=True, _accel=False)

    a = en_slow.uniform(*seqarr)
    b = en_fast.uniform(*seqarr)

    assert_array_equal(a, b)

def test_accel(seqarr):
    en_slow = sd.EnergeticsBasic(_tightloop=False, _accel=False)
    en_fast = sd.EnergeticsBasic(_accel=True)

    a = en_slow.uniform(*seqarr)
    b = en_fast.uniform(*seqarr)

    assert_array_equal(a, b)
