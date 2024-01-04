import numpy as np
try:
    from typing import Self, TypeVar
except ImportError:
    from typing_extensions import Self, TypeVar  # noqa: UP035
from typing import Union, Literal
from collections.abc import Sequence

__all__ = [
    'Energetics',
    'EndTypes',
    'EndArray',
    'PairSeqA',
    'endarray',
    'pairseqa',
    'nt',
    'lton',
    'wc',
    'tops',
]

EndTypes: TypeVar = Literal['DT', 'TD', 'S']

class Energetics:
    pass


class PairSeqA(np.ndarray):
    def __new__(cls, array: np.ndarray):
        obj = (4 * array[:, :-1] + array[:, 1:]).view(cls)
        return obj

    def revcomp(self) -> Self:
        return 4 * (3 - (self[:, ::-1] % 4)) + 3 - (self[:, ::-1] // 4)

    def tolist(self) -> Self:
        st = ["a", "c", "g", "t"]
        return [
            "".join([st[x // 4] for x in y] + [st[y[-1] % 4]]) for y in self
        ]

    def __repr__(self):
        return f"{self.tolist()!r}"

pairseqa = PairSeqA

class EndArray(np.ndarray):
    """
    A class for arrays full ends (of type adjacent+end+wc-adjacent-of-complementary-end).

    At present, it also handles adjacent+end style ends, but self.end and
    self.comp will return bogus information. It eventually needs to be split up
    into two classes in order to deal with this problem.
    """

    def __new__(cls, array: Union[Sequence[str], np.ndarray], endtype: EndTypes):
        if isinstance(array[0], str):
            array = np.array(
                [[nt[x] for x in y] for y in array], dtype=np.uint8)
        obj = np.asarray(array, dtype=np.uint8).view(cls)
        obj.endtype = endtype
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.endtype = getattr(obj, 'endtype', None)

    @property
    def ends(self) -> Self:
        if self.endtype == 'DT':
            return self[:, :-1]
        elif self.endtype == 'TD':
            return self[:, 1:]
        elif self.endtype == 'S':
            return self[:, :]

    @property
    def comps(self) -> Self:
        if self.endtype == 'DT':
            return (3 - self)[:, ::-1][:, :-1]
        elif self.endtype == 'TD':
            return (3 - self)[:, ::-1][:, 1:]
        elif self.endtype == 'S':
            return (3 - self)[:, ::-1][:, :]

    @property
    def fcomps(self):
        return (3-self)[:, ::-1]
        
    def concat(a1, a2):
        assert a1.endtype == a2.endtype

        return EndArray(
            np.concatenate((np.array(a1), np.array(a2)), axis=0),
            a1.endtype)

    def _get_adjs(self):
        if self.endtype == 'DT':
            return self[:, 0]
        elif self.endtype == 'TD':
            return self[:, -1]

    def _get_cadjs(self):
        if self.endtype == 'DT':
            return (3 - self)[:, -1]
        elif self.endtype == 'TD':
            return (3 - self)[:, 0]

    def __len__(self):
        return self.shape[0]

    def _get_endlen(self): # FIXME: not valid for S
        return self.shape[1] - 1

    def _get_seqlen(self):
        return self.shape[1]

    def append(s1, s2):
        assert s1.endtype == s2.endtype
        n = np.vstack((s1, s2)).view(EndArray)
        n.endtype = s1.endtype
        return n

    def __repr__(self):
        return f"<endarray ({len(self)}): type {self.endtype}; {self.tolist()!r}>"

    def tolist(self):
        st = ["a", "c", "g", "t"]
        return ["".join([st[x] for x in y]) for y in self]

    endlen = property(_get_endlen)
    seqlen = property(_get_seqlen)  # Added for new code compatibility
    adjs = property(_get_adjs)
    cadjs = property(_get_cadjs)
    strings = property(tolist)

endarray = EndArray

nt = {'a': 0, 'c': 1, 'g': 2, 't': 3, 'A': 0, 'C': 1, 'G': 2, 'T': 3}

lton = {
    'a': [0],
    'b': [1, 2, 3],
    'c': [1],
    'd': [0, 2, 3],
    'g': [2],
    'h': [0, 1, 3],
    'k': [2, 3],
    'm': [0, 1],
    'n': [0, 1, 2, 3],
    's': [1, 2],
    't': [3],
    'v': [0, 1, 2],
    'w': [0, 3]
}

wc = {
    'a': 't',
    'b': 'v',
    'c': 'g',
    'd': 'h',
    'g': 'c',
    'h': 'd',
    'k': 'm',
    'm': 'k',
    'n': 'n',
    's': 's',
    't': 'a',
    'v': 'b',
    'w': 'w'
}

def tops(s):
    return 4 * s[:, :-1] + s[:, 1:]
