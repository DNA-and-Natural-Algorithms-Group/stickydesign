import numpy as np
from math import ceil


import logging
LOGGER = logging.getLogger(__name__)


class BasicChooser():
    def __init__(self, energetics, interaction, wiggle=0.0):
        self._energetics = energetics
        self._target = interaction
        self._wiggle = wiggle

    def choose(self, availends, currentends=None):
        ddiff = np.abs(self._energetics.gse(availends) - self._target)
        choices = np.flatnonzero(ddiff <= np.amin(ddiff)+self._wiggle)
        return availends[choices[np.random.randint(0, len(choices))]]



def deviation_score(all_ends, all_energetics, devmethod='dev'):

    if devmethod == 'dev':
        return np.sqrt(
            np.sum(
                np.var(
                    [
                        np.concatenate(
                            tuple(en.gse(ends) for ends in all_ends))
                        for en in all_energetics
                    ],
                    axis=1)))
    elif devmethod == 'max':
        return np.max([np.ptp(np.concatenate(
            tuple(en.gse(ends) for ends in all_ends)))
                       for en in all_energetics])


class MultiModelChooser:

    def __init__(self, all_energetics,
                 target_vals=None,
                 init_wigglefraction=1,
                 next_wigglefraction=0.1,
                 devmethod='dev'):
        self._tv = target_vals
        self._ae = all_energetics
        self._iwf = init_wigglefraction
        self._nwf = next_wigglefraction
        self._dm = devmethod

    def choose(self, availends, currentends):

        if (currentends is None or len(currentends) == 0) and not self._tv:
            # Starting out, we need to choose an initial end, and use that
            # to choose our target values.
            availfiltered = availends
            dev = np.std(
                [en.gse(availends) for en in self._ae],
                axis=0)
            choices = np.argsort(dev)
            choice = choices[np.random.randint(
                0, max(1, ceil(self._iwf * len(availends))))]
            return availends[choice]
        else:
            if not self._tv:  # NOQA
                self._tv = [
                    en.gse(currentends[0:1])
                    for en in self._ae
                ]
                LOGGER.debug("TVALS {}".format(self._tv))
            availfiltered = availends
            if self._dm == 'dev':
                dev = np.sqrt(
                    np.sum(
                        np.array([
                            en.gse(availfiltered) - target_val
                            for en, target_val in zip(self._ae, self._tv)
                        ])**2,
                        axis=0))
            elif self._dm == 'max':
                dev = np.max(np.abs(
                        np.array([
                            en.gse(availfiltered) - target_val
                            for en, target_val in zip(self._ae, self._tv)
                        ])), axis=0)
            choices = np.argsort(dev)
            choice = choices[np.random.randint(
                0, max(1, ceil(len(choices) * self._nwf)))]
            LOGGER.debug("Chose {}: {} from {}: {} / {}".format(
                availfiltered[choice:choice+1].tolist(),
                np.nonzero(choices == choice),
                len(availfiltered),
                dev[choice],
                np.concatenate([en.gse(availfiltered[choice:choice+1]) for en,
                                tv in zip(self._ae, self._tv)])))
                         
            return availfiltered[choice]



