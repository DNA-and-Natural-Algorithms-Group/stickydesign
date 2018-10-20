from .seqclasses import _genvals, EndPairArrayTD, EndPairArrayDT, BoolSeqArray
from .energetics_daoe import EnergeticsDAOEC
from .filters import SeqFilter
from .choosers import BasicChooser

import numpy as np

def genpre(endtype, endlength, energetics=None, seqfilter=None, interaction=None,
           oldends=None):
    if energetics is None:
        energetics = EnergeticsDAOEC(endlength)

    epc = {'TD': EndPairArrayTD, 'DT': EndPairArrayDT}[endtype]

    availends = epc(_genvals(BoolSeqArray('n' * (endlength + 2))))
        
    if interaction is None:
        if oldends is not None:
            interaction = np.mean(energetics.gse(oldends))
        else:
            interaction = np.median(energetics.gse(availends))

    if seqfilter is None:
        seqfilter = SeqFilter(energetics, interaction)

    availends = seqfilter.filterspace(availends)

    if oldends is not None:
        availends = seqfilter.filterseqs(availends, oldends)
        
    return availends, interaction, seqfilter
    

def easyends(endtype, endlength, energetics=None, number=None,
             seqfilter=None, chooser=None, preavail=None, interaction=None,
             sfopts={}):

    if energetics is None:
        energetics = EnergeticsDAOEC(endlength)

    if preavail is None:
        epc = {'TD': EndPairArrayTD, 'DT': EndPairArrayDT}[endtype]

        availends = epc(_genvals(BoolSeqArray('n' * (endlength + 2))))
        
        if interaction is None:
            interaction = np.median(energetics.gse(availends))

        if seqfilter is None:
            seqfilter = SeqFilter(energetics, interaction, **sfopts)
        
        availends = seqfilter.filterspace(availends)
    else:
        availends = preavail.copy()
        
    if chooser is None:
        chooser = BasicChooser(energetics, interaction, wiggle=0.4)

    

    curends = None
    while True:
        newend = chooser.choose(availends, curends)
        if curends is None:
            curends = newend[None, :]
        else:
            curends = np.concatenate((curends, newend[None, :]),
                                     0).view(newend.__class__)
        availends = seqfilter.filterseqs(availends, newend[None, :])
        if len(availends) == 0:
            break
        elif number and len(curends) >= number:
            break

    return curends
