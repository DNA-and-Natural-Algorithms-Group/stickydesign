import pytest
import stickydesign as sd
import numpy.testing as npt
import numpy as np

def test_easyends_gen_s():
    ea = sd.easyends('S', 6, 8)
    assert ea.seqlen == 6
    assert ea.endlen == 6
    assert len(ea) == 8

    me = sd.EnergeticsBasic().matching_uniform(ea)
    assert np.all(me < 5.2)
    assert np.all(me > 4.8)
    


def test_easyends_gen_dt():
    energetics = sd.EnergeticsDAOE()
    ea = sd.easyends('DT', 6, 8, energetics=energetics)
    assert ea.seqlen == 8
    assert ea.endlen == 7
    assert len(ea) == 8

    me = energetics.matching_uniform(ea)
    assert np.all(me < 7.4)
    assert np.all(me > 7.0)
    


def test_easyends_gen_td():
    energetics = sd.EnergeticsDAOE()
    ea = sd.easyends('TD', 6, 8, energetics=energetics)
    assert ea.seqlen == 8
    assert ea.endlen == 7
    assert len(ea) == 8

    me = energetics.matching_uniform(ea)
    assert np.all(me < 7.1)
    assert np.all(me > 6.7)
    
def test_easyends_gen_td_no_energetics_specified():
    ea = sd.easyends('TD', 5, 5, interaction=6.0)
    assert ea.seqlen == 7
    assert ea.endlen == 6
    assert len(ea) == 5

    me = sd.EnergeticsDAOE().matching_uniform(ea)
    assert np.all(me < 6.3)
    assert np.all(me > 5.7)
    