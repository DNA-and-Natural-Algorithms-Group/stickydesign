import numpy as np
from .seqclasses import SeqArray
__version__ = None
from . import newparams as p


class EnergeticsBasic(object):
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
        info = {
            'enclass': self.__class__,
            'temperature': self.temperature,
            'coaxparams': self.coaxparaminfo,
            'danglecorr': self.danglecorr,
            'singlepair': self.singlepair,
            'version': __version__
        }
        return info

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, val):
        self._temperature = val
        self._setup_params(val)

    def __str__(self):
        return "EnergeticsBasic(" + \
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

    def gse(self, seqs1, seqs2=None):
        if seqs2 is None:
            return self.gse_matching(seqs1)
        else:
            return self.gse_all(seqs1, seqs2)
                    
    def gse_matching(self, seqs: SeqArray):
        # In both cases here, the energy we want is the NN binding energy of
        # each stack,
        return -(np.sum(self.nndG[seqs.nnseq], axis=-1) + self.initdG)

    def gse_all(self, seqs1, seqs2, forcemulti=False, debug=False):
        assert seqs1.shape[-1] == seqs2.shape[-1]
        if (seqs1.ndim == 2 and seqs2.ndim == 2
                and seqs1.shape[:-1] != seqs2.shape[:-1]):
            seqs1 = seqs1[:, None, :]
        if forcemulti:
            if seqs1.ndim == 1:
                seqs1 = seqs1[None, :]
            if seqs2.ndim == 1:
                seqs2 = seqs2[None, :]
            seqs1 = seqs1[:, None, :]

        s1 = seqs1.nnseq
        s2 = seqs2.nnseq
        l = s1.shape[-1]

        # s2r is revcomp pairseq of s2.
        s2r = s2.rc

        alloffset_max = np.zeros(s1.shape[:-1])
        # store for max binding at any offset

        for offset in range(-l + 1, l):
            if offset > 0:
                # Energies of matching stacks, zero otherwise. Can be used
                # to check match.
                ens = (s1[..., :-offset] == s2r[..., offset:]) * (
                    -self.nndG[s1[..., :-offset]])
                ltmm = -self.ltmmdG_5335[s1[..., :-offset] * 16
                                         + s2r[..., offset:]]
                rtmm = -self.rtmmdG_5335[s1[..., :-offset] * 16
                                         + s2r[..., offset:]]
                intmm = -self.intmmdG_5335[s1[..., :-offset] * 16
                                           + s2r[..., offset:]]
            elif offset == 0:
                ens = (s1 == s2r) * (-self.nndG[s1])
                ltmm = np.zeros_like(ens)
                rtmm = np.zeros_like(ens)
                intmm = np.zeros_like(ens)
                ltmm = -self.ltmmdG_5335[s1[..., :] * 16 + s2r[..., :]]
                rtmm = -self.rtmmdG_5335[s1[..., :] * 16 + s2r[..., :]]
                intmm = -self.intmmdG_5335[s1[..., :] * 16 + s2r[..., :]]
            else:  # offset < 0
                ens = (s1[..., -offset:] == s2r[..., :offset]) * (
                    -self.nndG[s1[..., -offset:]])
                ltmm = -self.ltmmdG_5335[s1[..., -offset:] * 16
                                         + s2r[..., :offset]]
                rtmm = -self.rtmmdG_5335[s1[..., -offset:] * 16
                                         + s2r[..., :offset]]
                intmm = -self.intmmdG_5335[s1[..., -offset:] * 16
                                           + s2r[..., :offset]]
            bindmax = np.zeros(ens.shape[:-1])
            if debug:
                print(offset, ens.view(np.ndarray), ltmm, rtmm, intmm)
            for endi in np.ndindex(ens.shape[:-1]):
                acc = 0
                for i in range(0, ens.shape[-1]):
                    if ens[endi][i] != 0:
                        # we're matching. add the pair to the accumulator
                        acc += ens[endi][i]
                    elif rtmm[endi][i] != 0 and i > 0 and ens[endi][i - 1] > 0:
                        # we're mismatching on the right: see if
                        # right-dangling is highest binding so far,
                        # and continue, adding intmm to accumulator.
                        # Update: we only want to do this if the last
                        # nnpair was bound, because otherwise, we
                        # can't have a "right" mismatch.
                        if acc + rtmm[endi][i] > bindmax[endi]:
                            bindmax[endi] = acc + rtmm[endi][i]
                        acc += intmm[endi][i]
                    elif ltmm[endi][i] != 0 and i < ens.shape[-1] - 1:
                        # don't do this for the last pair we're mismatching on
                        # the left: see if our ltmm is stronger than our
                        # accumulated binding+intmm. If so, reset to ltmm and
                        # continue as left-dangling, or reset to 0 if ltmm+next
                        # is weaker than next dangle,or next is also a mismatch
                        # (fixme: good idea?). If not, continue as internal
                        # mismatch.
                        if (not self.singlepair) and (
                                ltmm[endi][i] > acc + intmm[endi][i]) and (
                                    ens[endi][i + 1] > 0):
                            acc = ltmm[endi][i]
                        elif (self.singlepair) and (ltmm[endi][i] >
                                                    acc + intmm[endi][i]):
                            acc = ltmm[endi][i]
                        else:
                            acc += intmm[endi][i]
                    else:  # we're at a loop. Add stuff.
                        acc -= p.looppenalty
                bindmax[endi] = max(bindmax[endi], acc)

                if debug:
                    print(alloffset_max, bindmax)
            alloffset_max = np.maximum(alloffset_max, bindmax)
        return alloffset_max - self.initdG


class EnergeticsBasicC(EnergeticsBasic):
    """A wrapper around EnergeticsBasic that implements caching , using 1d
    storage array.  This will be identical in use to EnergeticsBasic,
    but will store results.  The exception here is that seqlen (the
    sequence length) must be prespecified.
    """

    def __init__(self, seqlen, *args, **kwargs):
        EnergeticsBasic.__init__(self, *args, **kwargs)
        self._store = np.ma.masked_all((4**(seqlen*2),))
        self._multarS = 4**np.arange(seqlen-1, -1, -1)
        self._multarB = 4**seqlen * self._multarS
        self.seqlen = seqlen

    def gse_all(self, seqs1, seqs2, forcemulti=False):
        assert seqs1.shape[-1] == seqs2.shape[-1] == self.seqlen
        if (seqs1.ndim == 2 and seqs2.ndim == 2
                and seqs1.shape[:-1] != seqs2.shape[:-1]):
            seqs1 = seqs1[:, None, :]
            seqs2 = seqs2[None, :, :]
            multi = True
        elif forcemulti:
            if seqs1.ndim == 1:
                seqs1 = seqs1[None, :]
            if seqs2.ndim == 1:
                seqs2 = seqs2[None, :]
            seqs1 = seqs1[:, None, :]
            seqs2 = seqs2[None, :, :]
            multi = True
        else:
            multi = False
            if seqs1.ndim == 1 and seqs2.ndim == 2:
                seqs1 = np.tile(seqs1, (len(seqs2), 1))

        # Get the indexes
        indexes = np.sum(seqs1 * self._multarB + seqs2 * self._multarS, axis=-1)

        todo = self._store.mask[indexes].nonzero()

        if multi and len(seqs1[todo[0]]) > 0:
            self._store[indexes[todo]] = EnergeticsBasic.gse_all(self,
                                                                 seqs1[todo[0], 0, :],
                                                                 seqs2[0, todo[1], :])
        elif not multi and len(seqs1[todo]) > 0:
            self._store[indexes[todo]] = EnergeticsBasic.gse_all(self,
                                                                 seqs1[todo],
                                                                 seqs2[todo])            

        assert np.all(~self._store[indexes].mask)
            
        return self._store[indexes].filled(np.nan)
