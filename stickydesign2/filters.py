from .seqclasses import SeqPairArray
import numpy as np


class SeqFilter(object):

    def __init__(self, energymodel, targetint,
                 maxdeviation=0.05, maxspurious=0.5,
                 maxendendspurious=None,
                 maxtype='fractional'):
        """
Create a SeqFilter instance.

Parameters
----------
energymodel : EnergyModel
    the energy model to use.  This can be changed later with
    the energymodel attribute.

targetint : float
    the target interaction (usually in kcal/mol).  This can be
    changed later with the targetint attribute.

maxdeviation : float, optional
    the maximum deviation allowed for an end from the target
    interaction.  If maxtype = 'fractional', then this is interpreted
    as a fraction of targetint above or below targetint. If maxtype =
    'relative' or 'absolute', this is interpreted as a kcal/mol value
    above or below targetint.
    (default 0.05)

maxspurious : float, optional
    the maximum spurious interaction allowed between an end and other
    ends / complements.  If maxtype = 'fractional', then this is
    interpreted as a fraction of targetint.  If maxtype = 'absolute',
    this is interpreted as an absolute kcal/mol threshold.  If
    maxtype = 'relative', this is interpreted as a kcal/mol amount
    below targetint.
    (default 0.5)

maxendendspurious : float, optional
    if None, then maxspurious will be used for everything.  If specified,
    then maxspurious will only apply to interactions between ends and
    the complements of other ends, while maxendendspurious will apply to
    interactions between ends and other ends.  This could be useful if,
    for example, your system design is such that ends can't easily interact
    with other ends.
    (default None)

maxtype : {'relative', 'absolute', 'fractional'}, optional
    determines how the max parameters are to be interpreted.  If
    'fractional', each is considered as a fraction of the targetint.
    If 'relative' or 'absolute', all are considered to be in kcal/mol.
    (default 'fractional')
        """
        self.energymodel = energymodel
        self._targetint = targetint
        self._maxdeviation = maxdeviation
        self._maxspurious = maxspurious
        self._maxendendspurious = maxendendspurious
        self._maxtype = maxtype
        self._recalcparams()

    def _recalcparams(self):
        """
        Recalculate parameters (maxdeviation, maxspurious, 
        maxendendspurious) on the basis of maxtype and targetint.

        This is called on initialization and whenever any parameter
        is changed.  You should never need to call it manually.
        """

        if self._maxendendspurious is None:
            mees = self._maxspurious
        
        if self._maxtype == 'fractional':
            self._rmd = self._maxdeviation * self._targetint
            self._rms = self._maxspurious * self._targetint
            self._rmees = mees * self._targetint
        elif self._maxtype == 'absolute':
            self._rmd = self._maxdeviation
            self._rms = self._maxspurious
            self._rmees = mees
        elif self._maxtype == 'relative':
            self._rmd = self._maxdeviation
            self._rms = self._targetint - self._maxspurious
            self._rmees = self._targetint - mees
        else:
            raise ValueError("{} not a recognized maxtype".format(
                self._maxtype))

    def filterspace(self, seqpairs):
        """
Filter a SeqPairArray down to only seqpairs that could satisfy the
filter's parameters: ensure that matching (seq-pair) interactions are
within the maxdeviation, and self (seq-seq, pair-pair) interactions are
within maxspurious or maxendendspurious.
        
Parameters
----------

seqpairs : SeqPairArray
    the seqpairs to filter.

Returns
-------
out : SeqPairArray
    the filtered seqpairs
        """
        ma = self.energymodel.gse(seqpairs)
        ss = self.energymodel.gse_all(seqpairs.seqs, seqpairs.seqs)
        cc = self.energymodel.gse_all(seqpairs.comps, seqpairs.comps)
        filt = ((np.abs(ma-self._targetint) < self._rmd) &
                (ss < self._rmees) & (cc < self._rmees))
        f1 = seqpairs[filt]

        # now filter poly-G:
        g4 = np.zeros(f1.shape[0])
        for w in range(0, (f1.shape[1] - 3)):
            g4 += (np.sum(
                np.array(f1[:, w:(w + 4)] == [2, 2, 2, 2]), axis=1) == 4)
            g4 += (np.sum(
                np.array(f1[:, w:(w + 4)] == [1, 1, 1, 1]), axis=1) == 4)
        return f1[g4 == 0]

    def filterseqs(self, avail, new):
        """
Filter a SeqPairArray, given the addition of new seqs: filter avail
by the interactions each seqpair has with the seqpair/seqpairs in new.

Parameters
----------

avail : SeqPairArray
    The seqpairs to be filtered.

new : SeqPairArray
    New seqs (seqs that have not yet been used to filter 
    available ends) that should be used to filter avail.

Returns
-------
out : SeqPairArray
    The filtered seqpairs that are still available.
        """

        ss = self.energymodel.gse_all(new.seqs, avail.seqs, forcemulti=True)
        sc = self.energymodel.gse_all(new.seqs, avail.comps, forcemulti=True)
        cs = self.energymodel.gse_all(new.comps, avail.seqs, forcemulti=True)
        cc = self.energymodel.gse_all(new.comps, avail.comps, forcemulti=True)

        filt = ((ss < self._rmees) &
                (cc < self._rmees) &
                (sc < self._rms) &
                (cs < self._rms)).all(axis=0)

        return avail[filt]
