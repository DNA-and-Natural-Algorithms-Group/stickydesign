import numpy as np

_REVBITS4 = np.array(
    [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15], dtype=np.uint8)

_BSA_TO_SA = np.ma.masked_array(
    data=[9, 0, 1, 9, 2, 9, 9, 9, 3] + [9] * 7,
    mask=[True, False, False, True, False, True, True, True, False] +
    [True] * 7,
    dtype='uint8')

_BSA_TO_SA_ARRAYS = np.array([
    np.array([0b1 & x, 0b10 & x, 0b100 & x,
              0b1000 & x]).nonzero()[0].astype(np.uint8) for x in range(0, 16)
])

def seqconcat(s1, s2):
    if isinstance(s1, SeqArrayBase) and isinstance(s2, SeqArrayBase):
        return np.concatenate((s1, s2), -1)

class SeqArrayBase(np.ndarray):
    """A base class for numpy-based sequence arrays.  This doesn't assume
    anything about how the array works, and can be used with isinstance
    to check.  Other classes should pull from this."""

    def __new__(cls, array):
        if isinstance(array, np.ndarray):
            return np.asarray(array, dtype=np.uint8).view(cls)            
        if type(array) is str:
            array = [array]
        if type(array[0]) is str:
            array = np.array(
                [[cls._nt[x] for x in y] for y in array], dtype=np.uint8)
        return np.asarray(array, dtype=np.uint8).view(cls)            

    @property
    def strs(self):
        if self.ndim == 2:
            return [''.join(self._l[x] for x in s) for s in self]
        elif self.ndim == 1:
            return ''.join(self._l[x] for x in self)    
    
    def __repr__(self):
        return "{}({})".format(type(self).__name__, repr(self.strs))

_NNRC = np.array([15, 11,  7,  3,
                  14, 10,  6,  2,
                  13,  9,  5,  1,
                  12,  8,  4,  0], dtype=np.uint8)
class NNSeqArray(np.ndarray):
    @property
    def _rccalc(self):
        return 4 * (3 - (self[..., ::-1] % 4)) + 3 - (self[..., ::-1] // 4)

    @property
    def rc(self):
        return _NNRC[self[..., ::-1]].view(self.__class__)

class SeqArray(SeqArrayBase):
    """A SeqArray that uses typical 0,1,2,3 = a,c,g,t coding"""
    _nt = {'a': 0, 'c': 1, 'g': 2, 't': 3}
    _l = {v: k for k, v in _nt.items()}

    @property
    def rc(self):
        return (3 - self)[..., ::-1].view(self.__class__)

    @property
    def nnseq(self):
        return (4 * self[..., :-1] + self[..., 1:]).view(NNSeqArray)

    @property
    def seqlen(self):
        return self.shape[-1]


class BoolSeqArray(SeqArrayBase):
    """A SeqArray that uses the 'bool' coding (a=0b0001, c=0b0010,
    n=0b1111, null=0b0000 etc)"""
    _nt = {
        'a': 1,
        'b': 14,
        'c': 2,
        'd': 13,
        'g': 4,
        'h': 11,
        'k': 12,
        'm': 3,
        'n': 15,
        's': 6,
        't': 8,
        'v': 7,
        'w': 9
    }
    _l = {v: k for k, v in _nt.items()}

    @property
    def rc(self):
        return _REVBITS4[self][..., ::-1].view(self.__class__)

    @property
    def seqarray(self):
        l = _BSA_TO_SA[self]
        if np.ma.is_masked(l):
            raise ValueError("There are ambiguous bases")
        return l.view(SeqArray)


class EndArray(SeqArray):
    pass


class SeqPairArray(SeqArray):
    @property
    def seqs(self):
        return self.view(SeqArray)

    @property
    def pairs(self):
        return self.rc.view(SeqArray)

    comps = pairs

class EndPairArray(EndArray):
    pass


class EndArrayDT(EndArray):
    pass


class EndArrayTD(EndArray):
    pass


class EndPairArrayDT(EndArray):
    @property
    def ends(self):
        """
Returns
-------
EndArrayDT
    an array of just the ends (+ their adjacents) from the array. 

        """
        return self[..., :-1].view(EndArrayDT)

    seqs = ends
    
    @property
    def comps(self):
        """
Returns
-------
EndArrayDT
    an array of just the complementary ends (+ their adjacents) 
    of the ends in the array.
        """
        return self.rc[..., :-1].view(EndArrayDT)

    pairs = comps


class EndPairArrayTD(EndArray):
    @property
    def ends(self):
        """
Returns
-------
EndArrayDT
    an array of just the ends (+ their adjacents) from the array. 

        """
        return self[..., 1:].view(EndArrayTD)

    seqs = ends
    
    @property
    def comps(self):
        """
Returns
-------
EndArrayDT
    an array of just the complementary ends (+ their adjacents) 
    of the ends in the array.
        """
        return self.rc[..., 1:].view(EndArrayTD)

    pairs = comps


def _cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def _genvals(template: BoolSeqArray):
    return _cartesian_product(*_BSA_TO_SA_ARRAYS[template[0]]).view(SeqArray)
