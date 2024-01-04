import pytest
from hypothesis import given
import hypothesis.strategies as st
from numpy.testing import assert_array_almost_equal

from stickydesign.endclasses import EndArray, PairSeqA

@given(st.lists(st.text(alphabet="acgtACGT", min_size=10, max_size=10), min_size=1, max_size=30))
def test_create_endarray(ealist):
    lowered_ealist = [x.lower() for x in ealist]
    ea = EndArray(ealist, "S")
    assert ea.tolist() == lowered_ealist
    assert ea.seqlen == 10
    psa = PairSeqA(ea)
    assert psa.tolist() == lowered_ealist

    revcomp_lowered_ealist = [x[::-1].translate(str.maketrans("acgt", "tgca")) for x in lowered_ealist]

    assert psa.revcomp().tolist() == revcomp_lowered_ealist

    assert repr(psa) == repr(lowered_ealist)