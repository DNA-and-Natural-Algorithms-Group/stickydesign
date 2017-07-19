from __future__ import division
import os
import pkg_resources
import numpy as np
from .endclasses import pairseqa, tops, endarray
from .version import __version__

# n.b.: All parameters arrays here are arranged to fit the binary
# representation used in stickydesign.  A, C, G, and T are given values of
# 0,1,2,3, thus taking up 2 bits, and these bits are then put in 5' to 3'
# order.  Thus, for example, 5 would correspond to 5'-CC-3', and 2 would
# correspond to 5'-AG-3'.

# yapf: disable

# dG and dS parameters from SantaLucia, 2004.
nndG37 = np.array([-1.00, -1.44, -1.28, -0.88,
                   -1.45, -1.84, -2.17, -1.28,
                   -1.30, -2.24, -1.84, -1.44,
                   -0.58, -1.30, -1.45, -1.00])
nndS = np.array([-0.0213, -0.0224, -0.0210, -0.0204,
                 -0.0227, -0.0199, -0.0272, -0.0210,
                 -0.0222, -0.0244, -0.0199, -0.0224,
                 -0.0213, -0.0222, -0.0227, -0.0213])

# Coaxial stacking parameter dGs at 37 C are from Protozanova 2004
# (doi:10.1016/j.jmb.2004.07.075), table 2.
coax_prot_tab2_dG37 = np.array([-1.04, -2.04, -1.29, -1.27,
                                -0.78, -1.97, -1.44, -1.29,
                                -1.66, -2.70, -1.97, -2.04,
                                -0.12, -1.66, -0.78, -1.04])

coax_prot_tab1_dG37 = np.array([-1.11, -1.81, -1.06, -1.34,
                                -0.55, -1.44, -0.91, -1.06,
                                -1.43, -2.17, -1.44, -1.81,
                                -0.19, -1.43, -0.55, -1.11])

# Coaxial stacking parameter dSs for Protozanova parameters are taken
# from Zhang, 2009 supplementary information, which discusses this
# formulaic approach.
coaxdS = 0.0027 / (1-310.15*0.0027) * coax_prot_tab2_dG37
coaxddS = coaxdS-nndS
coaxddG37 = coax_prot_tab2_dG37-nndG37
# correction term rather than absolute dG

# Coaxial stacking parameters from Pyshni, Ivanova 2002
coax_pyshni_dG37 = np.array([-1.22, -2.10, -1.19, -1.89,
                             -1.20, -1.96, -1.17, -1.35,
                             -1.46, -2.76, -1.25, -2.29,
                             -0.85, -2.21, -1.06, -1.93])

coax_pyshni_dS = 0.001 * np.array([-26.0, -30.1, -27.9, -29.0,
                                   -33.8, -21.4, -14.9, -15.3,
                                   -25.4, -35.6, -20.6, -35.6,
                                   -28.2, -45.6, -31.7, -48.3])
coax_pyshni_ddS = coax_pyshni_dS - nndS
coax_pyshni_ddG37 = coax_pyshni_dG37 - nndG37

# Coaxial stacking parameters from Peyret 2000. The nick here is on the
# reverse complement strand.  The TT and AG values are averages.
# These are currently unused.
coax_peyret_dG37 = np.array([-1.5, -1.9, -1.15, -2.2,
                             -1.7, -2.6, -0.8, -2.4,
                             -2.6, -3.3, -1.6, -3.0,
                             -1.6, -2.9, -1.7, -2.55])

coax_peyret_dS = 0.001 * np.array([-44.7, -27.8, -(25.4+56.9)/2.0, -59.2,
                                   -46.8, -52.0, -26.4, -33.5,
                                   -48.5, -31.4, -20.6, -55.5,
                                   -26.2, -34.1, -33.1, -(23.1+77.6)/2])

coax_peyret_ddS = coax_peyret_dS - nndS
coax_peyret_ddG37 = coax_peyret_dG37 - nndG37

dangle5dG37 = np.array([-0.51, -0.96, -0.58, -0.50,
                        -0.42, -0.52, -0.34, -0.02,
                        -0.62, -0.72, -0.56,  0.48,
                        -0.71, -0.58, -0.61, -0.10])
dangle5dH = np.array([0.2, -6.3, -3.7, -2.9,
                      0.6, -4.4, -4.0, -4.1,
                     -1.1, -5.1, -3.9, -4.2,
                     -6.9, -4.0, -4.9, -0.2])
dangle3dG37 = np.array([-0.12,  0.28, -0.01,  0.13,
                        -0.82, -0.31, -0.01, -0.52,
                        -0.92, -0.23, -0.44, -0.35,
                        -0.48, -0.19, -0.50, -0.29])
dangle3dH = np.array([-0.5,  4.7, -4.1, -3.8,
                      -5.9, -2.6, -3.2, -5.2,
                      -2.1, -0.2, -3.9, -4.4,
                      -0.7,  4.4, -1.6,  2.9])
intmmdG37 = np.array([0.61,  0.88,  0.14,  0.00,  # ij
                      0.77,  1.33,  0.00,  0.64,  # *k
                      0.02,  0.00, -0.13,  0.71,
                      0.00,  0.73,  0.07,  0.69,
                      0.43,  0.75,  0.03,  0.00,
                      0.79,  0.70,  0.00,  0.62,
                      0.11,  0.00, -0.11, -0.47,
                      0.00,  0.40, -0.32, -0.12,
                      0.17,  0.81, -0.25,  0.00,
                      0.47,  0.79,  0.00,  0.62,
                     -0.52,  0.00, -1.11,  0.08,
                      0.00,  0.98, -0.59,  0.45,
                      0.69,  0.92,  0.42,  0.00,
                      1.33,  1.05,  0.00,  0.97,
                      0.74,  0.00,  0.44,  0.43,
                      0.00,  0.75,  0.34,  0.68])
intmmdS = np.array([1.7,   4.6,  -2.3,   0.0,
                    14.6,  -4.4,   0.0,   0.2,
                    -2.3,   0.0,  -9.5,   0.9,
                    0.0,  -6.2,  -8.3, -10.8,
                    -4.2,   3.7,  -2.3,   0.0,
                    -0.6,  -7.2,   0.0,  -4.5,
                    -13.2,   0.0, -15.3, -11.7,
                    0.0,  -6.1,  -8.0, -15.8,
                    -9.8,  14.2,  -1.0,   0.0,
                    -3.8,   8.9,   0.0,   5.4,
                    3.2,   0.0, -15.8,  10.4,
                    0.0,  13.5, -12.3,  -8.4,
                    12.9,   8.0,   0.7,   0.0,
                    20.2,  16.4,   0.0,   0.7,
                    7.4,   0.0,   3.6,  -1.7,
                    0.0,   0.7,  -5.3,  -1.5])/1000.0
# go from cal/(molK) to kcal/(molK)

dangle5dS = (dangle5dH - dangle5dG37) / 310.15
dangle3dS = (dangle3dH - dangle3dG37) / 310.15
initdG37 = 1.96
initdS = 0.0057
tailcordG37 = 0.8
looppenalty = 3.6

# yapf: enable


class EnergeticsDAOE(object):
    """Energy functions based on several sources, primarily SantaLucia's 2004
paper, along with handling of dangles, tails, and nicks specifically for DX
tile sticky ends.

    """

    def __init__(self,
                 temperature=37,
                 mismatchtype='combined',
                 coaxparams=False,
                 singlepair=False,
                 danglecorr=True,
                 version=None,
                 enclass=None):
        self.coaxparams = coaxparams
        self.singlepair = singlepair
        self.danglecorr = danglecorr
        self.temperature = temperature
        self.mismatchtype = mismatchtype
        self.setup_params(temperature)

        try:
            dsb = pkg_resources.resource_stream(__name__,
                                                os.path.join(
                                                    'params',
                                                    'dnastackingbig.csv'))
        except:
            try:
                this_dir, this_filename = os.path.split(__file__)
                dsb = open(
                    os.path.join(this_dir, "params", "dnastackingbig.csv"))
            except IOError:
                raise IOError("Error loading dnastackingbig.csv")
        self.nndG37_full = -np.loadtxt(dsb, delimiter=',')
        dsb.close()

        if mismatchtype == 'max':
            self.uniform = lambda x, y: np.maximum(
                self.uniform_loopmismatch(x, y),
                self.uniform_danglemismatch(x, y))
        elif mismatchtype == 'loop':
            self.uniform = self.uniform_loopmismatch
        elif mismatchtype == 'dangle':
            self.uniform = self.uniform_danglemismatch
        elif mismatchtype == 'combined' or mismatchtype == 'new':
            self.uniform = self.uniform_combinedmismatch
        else:
            raise ValueError(
                "Mismatchtype {0} is not supported.".format(mismatchtype))

    @property
    def info(self):
        info = {'enclass': 'EnergeticsDAOE',
                'temperature': self.temperature,
                'mismatchtype': self.mismatchtype,
                'coaxparams': self.coaxparaminfo,
                'danglecorr': self.danglecorr,
                'singlepair': self.singlepair,
                'version': __version__}
        return info

    def __str__(self):
        return "EnergeticsDAOE(" + \
            ", ".join("{}={}".format(x, repr(y))
                      for x, y in self.info.items()) + \
            ")"

    def __repr__(self):
        return self.__str__()
    
    def setup_params(self, temperature=37):
        self.initdG = initdG37 - (temperature - 37) * initdS
        self.nndG = nndG37 - (temperature - 37) * nndS
        if self.coaxparams == 'protozanova' or self.coaxparams is True:
            self.coaxddG = coaxddG37 - (temperature - 37) * coaxddS
            self.coaxparaminfo = 'protozanova'
            self.coaxparams = True
        elif self.coaxparams == 'peyret':
            self.coaxddG = coax_peyret_ddG37 - (
                temperature - 37) * coax_peyret_ddS
            self.coaxparaminfo = self.coaxparams
            self.coaxparams = True
        elif self.coaxparams == 'pyshni':
            self.coaxddG = coax_pyshni_ddG37 - (
                temperature - 37) * coax_pyshni_ddS
            self.coaxparaminfo = self.coaxparams
            self.coaxparams = True
        elif not self.coaxparams:
            self.coaxddG = np.zeros_like(coaxddG37)
            self.coaxparaminfo = self.coaxparams
            self.coaxparams = False
        else:
            raise ValueError("Invalid coaxparams: {}".format(self.coaxparams),
                             self.coaxparams)
        self.dangle5dG = dangle5dG37 - (temperature - 37) * dangle5dS
        self.dangle3dG = dangle3dG37 - (temperature - 37) * dangle3dS
        self.intmmdG = intmmdG37 - (temperature - 37) * intmmdS

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
        ps = pairseqa(seqs)

        # In both cases here, the energy we want is the NN binding energy of
        # each stack,
        if seqs.endtype == 'DT':
            dcorr = np.zeros_like(self.dangle3dG[ps[:, 0]])
            if self.danglecorr:
                dcorr += -self.dangle3dG[ps[:, 0]] - self.dangle3dG[ps.revcomp(
                )[:, 0]]
            if self.coaxparams:
                dcorr += self.coaxddG[ps[:, 0]] + self.coaxddG[ps.revcomp()[:,
                                                                            0]]
        elif seqs.endtype == 'TD':
            dcorr = np.zeros_like(self.dangle5dG[ps[:, -1]])
            if self.danglecorr:
                dcorr += (-self.dangle5dG[ps[:, -1]] -
                          self.dangle5dG[ps.revcomp()[:, -1]])
            if self.coaxparams:
                dcorr += self.coaxddG[ps[:, -1]] + self.coaxddG[ps.revcomp()
                                                                [:, -1]]
        return -(np.sum(self.nndG[ps], axis=1) + self.initdG + dcorr)

    def uniform_loopmismatch(self, seqs1, seqs2):
        if seqs1.shape != seqs2.shape:
            if seqs1.ndim == 1:
                seqs1 = endarray(
                    np.repeat(np.array([seqs1]), seqs2.shape[0], 0),
                    seqs1.endtype)
            else:
                raise ValueError(
                    "Lengths of sequence arrays are not acceptable.")
        assert seqs1.endtype == seqs2.endtype
        endtype = seqs1.endtype

        endlen = seqs1.endlen
        plen = endlen - 1

        s1 = tops(seqs1)
        s2 = tops(seqs2)
        # TODO: replace this with cleaner code
        if endtype == 'DT':
            ps1 = seqs1[:, 1:-1] * 4 + seqs1[:, 2:]
            pa1 = seqs1[:, 0] * 4 + seqs1[:, 1]
            pac1 = (3 - seqs1[:, 0]) * 4 + seqs2[:, -1]
            ps2 = seqs2[:, ::-1][:, :-2] * 4 + seqs2[:, ::-1][:, 1:-1]
            pa2 = seqs2[:, 0] * 4 + seqs2[:, 1]
            pac2 = (3 - seqs2[:, 0]) * 4 + seqs1[:, -1]
        if endtype == 'TD':
            ps1 = seqs1[:, :-2] * 4 + seqs1[:, 1:-1]
            pa1 = seqs1[:, -2] * 4 + seqs1[:, -1]
            pac1 = seqs2[:, 0] * 4 + (3 - seqs1[:, -1])
            ps2 = seqs2[:, ::-1][:, 1:-1] * 4 + seqs2[:, ::-1][:, 2:]
            pa2 = seqs2[:, -2] * 4 + seqs2[:, -1]
            pac2 = (seqs1[:, 0]) * 4 + (3 - seqs2[:, -1])

        # Shift here is considering the first strand as fixed, and the second
        # one as shifting.  The shift is the offset of the bottom one in terms
        # of pair sequences (thus +2 and -1 instead of +1 and 0).
        en = np.zeros((ps1.shape[0], 2 * plen))
        for shift in range(-plen + 1, plen):
            en[:, plen + shift - 1] = np.sum(
                self.nndG37_full[ps1[:, max(shift, 0):plen + shift],
                                 ps2[:, max(-shift, 0):plen - shift]],
                axis=1)
        en[:, plen -
           1] = en[:, plen -
                   1] + self.nndG37_full[pa1, pac1] + self.nndG37_full[pa2,
                                                                       pac2]
        if endtype == 'DT':
            en[:, plen - 1] += (self.nndG37_full[pa1, pac1] > 0) * (
                +self.dangle3dG[s1[:, 0]] - self.coaxparams * self.
                coaxddG[s1[:, 0]]) + (self.nndG37_full[pa2, pac2] > 0) * (
                    +self.dangle3dG[s2[:, 0]] -
                    self.coaxparams * self.coaxddG[s2[:, 0]])  # sign reversed
        if endtype == 'TD':
            en[:, plen - 1] += +(self.nndG37_full[pa1, pac1] > 0) * (
                self.dangle5dG[s1[:, -1]] - self.coaxparams * self.
                coaxddG[s1[:, -1]]) + (self.nndG37_full[pa2, pac2] > 0) * (
                    self.dangle5dG[s2[:, -1]] -
                    self.coaxparams * self.coaxddG[s2[:, -1]])  # sign reversed
        return np.amax(en, 1) - self.initdG

    def uniform_danglemismatch(self, seqs1, seqs2, fast=True):
        if seqs1.shape != seqs2.shape:
            if seqs1.ndim == 1:
                seqs1 = endarray(
                    np.repeat(np.array([seqs1]), seqs2.shape[0], 0),
                    seqs1.endtype)
            else:
                raise ValueError(
                    "Lengths of sequence arrays are not acceptable.")
        assert seqs1.endtype == seqs2.endtype
        endtype = seqs1.endtype
        s1 = tops(seqs1)
        s2 = tops(seqs2)
        l = s1.shape[1]
        s2r = np.fliplr(np.invert(s2) % 16)
        s2r = s2r // 4 + 4 * (s2r % 4)
        m = np.zeros((s1.shape[0], 2 * np.sum(np.arange(2, l + 1)) + l + 1))
        r = np.zeros(m.shape[0])
        z = 0
        if endtype == 'TD':
            s1c = s1[:, 0:-1]
            s2rc = s2r[:, 1:]
            s1l = np.hstack(((4 * (s2r[:, 0] // 4) + s1[:, 0] // 4).reshape(
                -1, 1), s1))
            s2rl = np.hstack(
                (s2r, (4 * (s2r[:, -1] % 4) + s1[:, -1] % 4).reshape(-1, 1)))
        elif endtype == 'DT':
            s1c = s1[:, 1:]
            s2rc = s2r[:, 0:-1]
            s2rl = np.hstack(((4 * (s1[:, 0] // 4) + s2r[:, 0] // 4).reshape(
                -1, 1), s2r))
            s1l = np.hstack(
                (s1, (4 * (s1[:, -1] % 4) + s2r[:, -1] % 4).reshape(-1, 1)))
        for o in range(1, l - 1):
            zn = l - 1 - o
            m[:, z:z + zn] = (
                s1c[:, :-o] == s2rc[:, o:]
            ) * -self.nndG[s1c[:, :-o]]  # - for positive sign
            if endtype == 'DT':  # squish offset
                m[:, z] += (m[:, z] != 0) * (-self.nndG[s1[:, 0]] - tailcordG37
                                             + self.dangle3dG[s1[:, 0]]
                                             )  # - for positive sign
                m[:, z + zn - 1] += (m[:, z + zn - 1] != 0) * (
                    -self.nndG[s2[:, 0]] - tailcordG37 +
                    self.dangle3dG[s2[:, 0]])  # - for positive sign
            if endtype == 'TD':  # stretch offset
                m[:, z] += (m[:, z] != 0) * (-self.dangle3dG[s1c[:, -o]]
                                             )  # - for positive sign
                m[:, z + zn - 1] += (m[:, z + zn - 1] != 0) * (
                    -self.dangle3dG[s2[:, -o - 1]])  # - for positive sign
            z = z + zn + 2
            m[:, z:z + zn] = (
                s2rc[:, :-o] == s1c[:, o:]
            ) * -self.nndG[s2rc[:, :-o]]  # - for positive sign
            if endtype == 'DT':  # stretch offset
                m[:, z] += (m[:, z] != 0) * (-self.dangle5dG[s1c[:, o - 1]]
                                             )  # - for positive sign
                m[:, z + zn - 1] += (m[:, z + zn - 1] != 0) * (
                    -self.dangle5dG[s2[:, o]])  # - for positive sign
            if endtype == 'TD':  # squish offset
                m[:, z] += (m[:, z] != 0) * (
                    -self.nndG[s1[:, -1]] - tailcordG37 +
                    self.dangle5dG[s1[:, -1]])  # - for positive sign
                m[:, z + zn - 1] += (m[:, z + zn - 1] != 0) * (
                    -self.nndG[s2[:, -1]] - tailcordG37 +
                    self.dangle5dG[s2[:, -1]])  # - for positive sign
            z = z + zn + 2
        # The zero shift case
        m[:, z:z + l + 1] = -(
            (s1l == s2rl) * self.nndG[s1l])  # - for positive sign
        if endtype == 'DT':
            m[:, z] += (m[:, z] != 0) * (
                +self.dangle3dG[s1[:, 0]] -
                self.coaxparams * self.coaxddG[s1[:, 0]])  # sign reversed
            m[:, z + l] += (m[:, z + l] != 0) * (
                +self.dangle3dG[s2[:, 0]] -
                self.coaxparams * self.coaxddG[s2[:, 0]])  # sign reversed
        if endtype == 'TD':
            m[:, z] += +(m[:, z] != 0) * (
                self.dangle5dG[s1[:, -1]] -
                self.coaxparams * self.coaxddG[s1[:, -1]])  # sign reversed
            m[:, z + l] += +(m[:, z + l] != 0) * (
                self.dangle5dG[s2[:, -1]] -
                self.coaxparams * self.coaxddG[s2[:, -1]])  # sign reversed
        from ._stickyext import fastsub
        x = m
        fastsub(x, r)

        return r - self.initdG

    def uniform_combinedmismatch(self, seqs1, seqs2, debug=False):
        if seqs1.shape != seqs2.shape:
            if seqs1.ndim == 1:
                seqs1 = endarray(
                    np.repeat(np.array([seqs1]), seqs2.shape[0], 0),
                    seqs1.endtype)
            else:
                raise ValueError(
                    "Lengths of sequence arrays are not acceptable.")

        assert seqs1.endtype == seqs2.endtype
        endtype = seqs1.endtype

        s1 = tops(seqs1)
        s2 = tops(seqs2)
        l = s1.shape[1]

        # s2r is revcomp pairseq of s2.
        s2r = np.fliplr(np.invert(s2) % 16)
        s2r = s2r // 4 + 4 * (s2r % 4)

        alloffset_max = np.zeros(
            s1.shape[0])  # store for max binding at any offset

        if endtype == 'TD':
            s1_end = s1[:, 0:-1]  #
            s2_end_rc = s2r[:, 1:]
            s1l = np.hstack(((4 * (s2r[:, 0] // 4) + s1[:, 0] // 4).reshape(
                -1, 1), s1))
            s2rl = np.hstack(
                (s2r, (4 * (s2r[:, -1] % 4) + s1[:, -1] % 4).reshape(-1, 1)))
        elif endtype == 'DT':
            s1_end = s1[:, 1:]
            s2_end_rc = s2r[:, 0:-1]
            s2rl = np.hstack(((4 * (s1[:, 0] // 4) + s2r[:, 0] // 4).reshape(
                -1, 1), s2r))
            s1l = np.hstack(
                (s1, (4 * (s1[:, -1] % 4) + s2r[:, -1] % 4).reshape(-1, 1)))

        for offset in range(-l + 2, l - 1):
            if offset > 0:
                if endtype == 'TD':
                    # Energies of matching stacks, zero otherwise. Can be used
                    # to check match.
                    ens = (s1_end[:, :-offset] == s2_end_rc[:, offset:]) * (
                        -self.nndG[s1_end[:, :-offset]])
                    ens[:, 0] += (
                        ens[:, 0] != 0) * (-self.dangle3dG[s1_end[:, -offset]]
                                           )  # - for positive sign
                    ens[:, -1] += (
                        ens[:, -1] != 0) * (-self.dangle3dG[s2[:, -offset - 1]]
                                            )  # - for positive sign
                    ltmm = -self.ltmmdG_5335[s1_end[:, :-offset] * 16
                                             + s2_end_rc[:, offset:]]
                    rtmm = -self.rtmmdG_5335[s1_end[:, :-offset] * 16
                                             + s2_end_rc[:, offset:]]
                    intmm = -self.intmmdG_5335[s1_end[:, :-offset] * 16
                                               + s2_end_rc[:, offset:]]
                if endtype == 'DT':
                    ens = (s1_end[:, offset:] == s2_end_rc[:, :-offset]) * (
                        -self.nndG[s1_end[:, offset:]])
                    ens[:, 0] += (ens[:, 0] != 0) * (
                        -self.dangle5dG[s1_end[:, offset - 1]]
                    )  # - for positive sign
                    ens[:, -1] += (ens[:, -1] != 0) * (
                        -self.dangle5dG[s2[:, offset]])  # - for positive sign
                    ltmm = -self.ltmmdG_5335[s1_end[:, offset:] * 16
                                             + s2_end_rc[:, :-offset]]
                    rtmm = -self.rtmmdG_5335[s1_end[:, offset:] * 16
                                             + s2_end_rc[:, :-offset]]
                    intmm = -self.intmmdG_5335[s1_end[:, offset:] * 16
                                               + s2_end_rc[:, :-offset]]
            elif offset == 0:
                ens = (s1_end == s2_end_rc) * (-self.nndG[s1_end])
                if endtype == 'DT':
                    ens[:, 0] += (ens[:, 0] != 0) * (
                        +self.dangle3dG[s1[:, 0]] -
                        self.coaxparams * self.coaxddG[s1[:, 0]]) - (
                            s1l[:, 0] == s2rl[:, 0]
                        ) * self.nndG[s1l[:, 0]]  # sign reversed
                    ens[:, -1] += (ens[:, -1] != 0) * (
                        +self.dangle3dG[s2[:, 0]] -
                        self.coaxparams * self.coaxddG[s2[:, 0]]) - (
                            s1l[:, -1] == s2rl[:, -1]
                        ) * self.nndG[s1l[:, -1]]  # sign reversed
                if endtype == 'TD':
                    ens[:, 0] += +(ens[:, 0] != 0) * (
                        self.dangle5dG[s1[:, -1]] -
                        self.coaxparams * self.coaxddG[s1[:, -1]]) - (
                            s1l[:, 0] == s2rl[:, 0]
                        ) * self.nndG[s1l[:, 0]]  # sign reversed
                    ens[:, -1] += +(ens[:, -1] != 0) * (
                        self.dangle5dG[s2[:, -1]] -
                        self.coaxparams * self.coaxddG[s2[:, -1]]) - (
                            s1l[:, -1] == s2rl[:, -1]
                        ) * self.nndG[s1l[:, -1]]  # sign reversed
                ltmm = np.zeros_like(ens)
                rtmm = np.zeros_like(ens)
                intmm = np.zeros_like(ens)
                ltmm = -self.ltmmdG_5335[s1_end[:, :] * 16 + s2_end_rc[:, :]]
                rtmm = -self.rtmmdG_5335[s1_end[:, :] * 16 + s2_end_rc[:, :]]
                intmm = -self.intmmdG_5335[s1_end[:, :] * 16 + s2_end_rc[:, :]]
            else:  # offset < 0
                if endtype == 'TD':
                    ens = (s1_end[:, -offset:] == s2_end_rc[:, :offset]) * (
                        -self.nndG[s1_end[:, -offset:]])
                    ens[:, 0] += (ens[:, 0] != 0) * (
                        -self.nndG[s1[:, -1]] - tailcordG37 +
                        self.dangle5dG[s1[:, -1]])  # - for positive sign
                    ens[:, -1] += (ens[:, -1] != 0) * (
                        -self.nndG[s2[:, -1]] - tailcordG37 +
                        self.dangle5dG[s2[:, -1]])  # - for positive sign
                    ltmm = -self.ltmmdG_5335[s1_end[:, -offset:] * 16
                                             + s2_end_rc[:, :offset]]
                    rtmm = -self.rtmmdG_5335[s1_end[:, -offset:] * 16
                                             + s2_end_rc[:, :offset]]
                    intmm = -self.intmmdG_5335[s1_end[:, -offset:] * 16
                                               + s2_end_rc[:, :offset]]
                elif endtype == 'DT':
                    ens = (s1_end[:, :offset] == s2_end_rc[:, -offset:]) * (
                        -self.nndG[s1_end[:, :offset]])
                    ens[:, 0] += (ens[:, 0] != 0) * (
                        -self.nndG[s1[:, 0]] - tailcordG37 +
                        self.dangle3dG[s1[:, 0]])  # - for positive sign
                    ens[:, -1] += (ens[:, -1] != 0) * (
                        -self.nndG[s2[:, 0]] - tailcordG37 +
                        self.dangle3dG[s2[:, 0]])  # - for positive sign
                    ltmm = -self.ltmmdG_5335[s1_end[:, :offset] * 16
                                             + s2_end_rc[:, -offset:]]
                    rtmm = -self.rtmmdG_5335[s1_end[:, :offset] * 16
                                             + s2_end_rc[:, -offset:]]
                    intmm = -self.intmmdG_5335[s1_end[:, :offset] * 16
                                               + s2_end_rc[:, -offset:]]
            bindmax = np.zeros(ens.shape[0])
            if debug:
                print(offset, ens.view(np.ndarray), ltmm, rtmm, intmm)
            for e in range(0, ens.shape[0]):
                acc = 0
                for i in range(0, ens.shape[1]):
                    if ens[e, i] != 0:
                        # we're matching. add the pair to the accumulator
                        acc += ens[e, i]
                    elif rtmm[e, i] != 0:
                        # we're mismatching on the right: see if right-dangling
                        # is highest binding so far, and continue, adding intmm
                        # to accumulator.
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
                        acc -= looppenalty
                bindmax[e] = max(bindmax[e], acc)
            alloffset_max = np.maximum(alloffset_max, bindmax)
        return alloffset_max - self.initdG
