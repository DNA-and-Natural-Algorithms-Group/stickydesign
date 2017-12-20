from __future__ import division
import numpy as np
from .endclasses import pairseqa, tops, endarray, Energetics
from .version import __version__

from . import newparams as p


class EnergeticsBasic(Energetics):

    """Energy functions based on several sources, primarily SantaLucia's
       2004 paper.  This class uses the same parameters and algorithms
       as EnergeticsDAOE, bet does not make DX-specific assumptions.
       Instead, it assumes that each energy should simply be that of
       two single strands attaching/detaching, without consideration
       of nicks, stacking, or other effects related to the
       beginning/end of each sequence.  Dangles and tails are still
       included in mismatched binding calculations when appropriate.

       Relevant arguments:
       
       singlepair (bool, default False) --- treat single base pair pairings
       as possible.
       temperature (float in degrees Celsius, default 37) --- temperature
       to use for the model, in C.
    """
    
    def __init__(self,
                 temperature=37,
                 coaxparams=False,
                 singlepair=False,
                 danglecorr=True,
                 version=None,
                 enclass=None):
        self.coaxparams = coaxparams
        self.singlepair = singlepair
        self.danglecorr = danglecorr
        self.temperature = temperature

    @property
    def info(self):
        info = {'enclass': 'EnergeticsDAOE',
                'temperature': self.temperature,
                'coaxparams': self.coaxparaminfo,
                'danglecorr': self.danglecorr,
                'singlepair': self.singlepair,
                'version': __version__}
        return info

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, val):
        self._temperature = val
        self._setup_params(val)
    
    def __str__(self):
        return "EnergeticsDAOE(" + \
            ", ".join("{}={}".format(x, repr(y))
                      for x, y in self.info.items()) + \
            ")"

    def __repr__(self):
        return self.__str__()
    
    def _setup_params(self, temperature=37):
        self.initdG = p.initdG37 - (temperature - 37) * p.initdS
        self.nndG = p.nndG37 - (temperature - 37) * p.nndS
        if self.coaxparams == 'protozanova' or self.coaxparams is True:
            self.coaxddG = p.coaxddG37 - (temperature - 37) * p.coaxddS
            self.coaxparaminfo = 'protozanova'
            self.coaxparams = True
        elif self.coaxparams == 'peyret':
            self.coaxddG = p.coax_peyret_ddG37 - (
                temperature - 37) * p.coax_peyret_ddS
            self.coaxparaminfo = self.coaxparams
            self.coaxparams = True
        elif self.coaxparams == 'pyshni':
            self.coaxddG = p.coax_pyshni_ddG37 - (
                temperature - 37) * p.coax_pyshni_ddS
            self.coaxparaminfo = self.coaxparams
            self.coaxparams = True
        elif not self.coaxparams:
            self.coaxddG = np.zeros_like(p.coaxddG37)
            self.coaxparaminfo = self.coaxparams
            self.coaxparams = False
        else:
            raise ValueError("Invalid coaxparams: {}".format(self.coaxparams),
                             self.coaxparams)
        self.dangle5dG = p.dangle5dG37 - (temperature - 37) * p.dangle5dS
        self.dangle3dG = p.dangle3dG37 - (temperature - 37) * p.dangle3dS
        self.intmmdG = p.intmmdG37 - (temperature - 37) * p.intmmdS

        self.ltmmdG_5335 = np.zeros(256)
        self.rtmmdG_5335 = np.zeros(256)
        self.intmmdG_5335 = np.zeros(256)

        # Dumb setup. FIXME: do this cleverly
        for i in range(0, 4):
            for j in range(0, 4):
                for k in range(0, 4):
                    self.ltmmdG_5335[
                        i * 64 + j * 16 + k * 4 +
                        j] = self.dangle5dG[i * 4
                                            + j] + self.dangle3dG[(3 - j) * 4
                                                                  + (3 - k)]
                    self.rtmmdG_5335[
                        i * 64 + j * 16 + i * 4 +
                        k] = self.dangle3dG[i * 4
                                            + j] + self.dangle5dG[(3 - k) * 4
                                                                  + (3 - i)]
                    self.intmmdG_5335[i * 64 + j * 16 + k * 4 +
                                      j] = self.intmmdG[(3 - j) * 16
                                                        + (3 - k) * 4 + i]
                    self.intmmdG_5335[i * 64 + j * 16 + i * 4 +
                                      k] = self.intmmdG[i * 16
                                                        + j * 4 + (3 - k)]

    def matching_uniform(self, seqs):
        assert seqs.endtype == 'S'
        ps = pairseqa(seqs)

        # In both cases here, the energy we want is the NN binding energy of
        # each stack,
        return -(np.sum(self.nndG[ps], axis=1) + self.initdG)

    def uniform(self, seqs1, seqs2, debug=False):
        assert seqs1.endtype == seqs2.endtype
        assert seqs1.endtype == 'S'
        if seqs1.shape != seqs2.shape:
            if seqs1.ndim == 1:
                seqs1 = endarray(
                    np.repeat(np.array([seqs1]), seqs2.shape[0], 0),
                    seqs1.endtype)
            else:
                raise ValueError(
                    "Lengths of sequence arrays are not acceptable.")

        s1 = tops(seqs1)
        s2 = tops(seqs2)
        l = s1.shape[1]

        # s2r is revcomp pairseq of s2.
        s2r = np.fliplr(np.invert(s2) % 16)
        s2r = s2r // 4 + 4 * (s2r % 4)

        alloffset_max = np.zeros(
            s1.shape[0])  # store for max binding at any offset

        s1_end = s1[:, :]
        s2_end_rc = s2r[:, :]
        
        for offset in range(-l + 1, l):
            if offset > 0:
                    # Energies of matching stacks, zero otherwise. Can be used
                    # to check match.
                ens = (s1_end[:, :-offset] == s2_end_rc[:, offset:]) * (
                    -self.nndG[s1_end[:, :-offset]])
                ltmm = -self.ltmmdG_5335[s1_end[:, :-offset] * 16
                                         + s2_end_rc[:, offset:]]
                rtmm = -self.rtmmdG_5335[s1_end[:, :-offset] * 16
                                         + s2_end_rc[:, offset:]]
                intmm = -self.intmmdG_5335[s1_end[:, :-offset] * 16
                                           + s2_end_rc[:, offset:]]
            elif offset == 0:
                ens = (s1_end == s2_end_rc) * (-self.nndG[s1_end])
                ltmm = np.zeros_like(ens)
                rtmm = np.zeros_like(ens)
                intmm = np.zeros_like(ens)
                ltmm = -self.ltmmdG_5335[s1_end[:, :] * 16 + s2_end_rc[:, :]]
                rtmm = -self.rtmmdG_5335[s1_end[:, :] * 16 + s2_end_rc[:, :]]
                intmm = -self.intmmdG_5335[s1_end[:, :] * 16 + s2_end_rc[:, :]]
            else:  # offset < 0
                ens = (s1_end[:, -offset:] == s2_end_rc[:, :offset]) * (
                    -self.nndG[s1_end[:, -offset:]])
                ltmm = -self.ltmmdG_5335[s1_end[:, -offset:] * 16
                                         + s2_end_rc[:, :offset]]
                rtmm = -self.rtmmdG_5335[s1_end[:, -offset:] * 16
                                         + s2_end_rc[:, :offset]]
                intmm = -self.intmmdG_5335[s1_end[:, -offset:] * 16
                                           + s2_end_rc[:, :offset]]
            bindmax = np.zeros(ens.shape[0])
            if debug:
                print(offset, ens.view(np.ndarray), ltmm, rtmm, intmm)
            for e in range(0, ens.shape[0]):
                acc = 0
                for i in range(0, ens.shape[1]):
                    if ens[e, i] != 0:
                        # we're matching. add the pair to the accumulator
                        acc += ens[e, i]
                    elif rtmm[e, i] != 0 and i > 0 and ens[e, i-1] > 0:
                        # we're mismatching on the right: see if
                        # right-dangling is highest binding so far,
                        # and continue, adding intmm to accumulator.
                        # Update: we only want to do this if the last
                        # nnpair was bound, because otherwise, we
                        # can't have a "right" mismatch.
                        if acc + rtmm[e, i] > bindmax[e]:
                            bindmax[e] = acc + rtmm[e, i]
                        acc += intmm[e, i]
                    elif ltmm[e, i] != 0 and i < ens.shape[1] - 1:
                        # don't do this for the last pair we're mismatching on
                        # the left: see if our ltmm is stronger than our
                        # accumulated binding+intmm. If so, reset to ltmm and
                        # continue as left-dangling, or reset to 0 if ltmm+next
                        # is weaker than next dangle,or next is also a mismatch
                        # (fixme: good idea?). If not, continue as internal
                        # mismatch.
                        if (not self.singlepair) and (ltmm[e, i] >
                                                      acc + intmm[e, i]) and (
                                                          ens[e, i + 1] > 0):
                            acc = ltmm[e, i]
                        elif (self.singlepair) and (ltmm[e, i] >
                                                    acc + intmm[e, i]):
                            acc = ltmm[e, i]
                        else:
                            acc += intmm[e, i]
                    else:  # we're at a loop. Add stuff.
                        acc -= p.looppenalty
                bindmax[e] = max(bindmax[e], acc)
            if debug:
                print(alloffset_max, bindmax)
            alloffset_max = np.maximum(alloffset_max, bindmax)
        return alloffset_max - self.initdG
