import numpy as np
from typing_extensions import TypeAlias  # noqa: UP035
from typing import Union, Literal, List, cast, Dict, Any  # noqa: UP035
from collections.abc import Sequence
from abc import ABC, abstractmethod, abstractproperty

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

EndTypes: TypeAlias = Literal['DT', 'TD', 'S']  # noqa: UP040


class PairSeqA(np.ndarray):
    def __new__(cls, array: np.ndarray):
        obj = (4 * array[:, :-1] + array[:, 1:]).view(cls)
        return obj

    def revcomp(self) -> 'PairSeqA':
        return cast(PairSeqA, 4 * (3 - (self[:, ::-1] % 4)) + 3 - (self[:, ::-1] // 4))

    def tolist(self) -> List[str]:  # noqa: UP006
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

    FIXME: above is outdated, support should be fine.
    """

    endtype: EndTypes

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
    def ends(self) -> 'EndArray':
        if self.endtype == 'DT':
            x = cast(EndArray, self[:, :-1])
            # x.endtype = 'DTe'
            return x
        elif self.endtype == 'TD':
            x = cast(EndArray, self[:, 1:])
            # x.endtype = 'TeD'
            return x
        elif self.endtype in ['S', 'DTe', 'TeD']:
            return cast(EndArray, self[:, :])
        else:
            raise ValueError("Invalid endtype")

    @property
    def comps(self) -> 'EndArray':
        if self.endtype == 'DT':
            x = cast(EndArray, (3 - self)[:, ::-1][:, :-1])
            # x.endtype = 'DTe'
            return x
        elif self.endtype == 'TD':
            x = cast(EndArray, (3 - self)[:, ::-1][:, 1:])
            # x.endtype = 'TeD'
            return x
        elif self.endtype == 'S':
            return cast(EndArray, (3 - self)[:, ::-1][:, :])
        elif self.endtype in ['DTe', 'TeD']:
            raise ValueError("Cannot get comps of non-full TD or DT (TeD or DTe) type.")
        else:
            raise ValueError("Invalid endtype")

    @property
    def fcomps(self):
        return (3-self)[:, ::-1]
        
    def concat(a1, a2):
        assert a1.endtype == a2.endtype

        return EndArray(
            np.concatenate((np.array(a1), np.array(a2)), axis=0),
            a1.endtype)

    @property
    def adjs(self) -> 'EndArray':
        if self.endtype == 'DT':
            return cast(EndArray, self[:, 0])
        elif self.endtype == 'TD':
            return cast(EndArray, self[:, -1])
        elif self.endtype in ['S', 'DTe', 'TeD']:
            raise ValueError(f"End type {self.endtype} does not include adjacent bases.")
        else:
            raise ValueError(f"Invalid endtype {self.endtype}")

    @property
    def cadjs(self) -> 'EndArray':
        if self.endtype == 'DT':
            return cast(EndArray, (3 - self)[:, -1])
        elif self.endtype == 'TD':
            return cast(EndArray, (3 - self)[:, 0])
        elif self.endtype in ['S', 'DTe', 'TeD']:
            raise ValueError(f"End type {self.endtype} does not include adjacent bases.")
        else:
            raise ValueError(f"Invalid endtype {self.endtype}")

    def __len__(self):
        return self.shape[0]

    @property
    def endlen(self) -> int: # FIXME: not valid for S
        if self.endtype in ['DT', 'TD']:
            return self.shape[1] - 1
        elif self.endtype in ['S', 'DTe', 'TeD']:
            return self.shape[1]
        else:
            raise ValueError(f"Invalid endtype {self.endtype}")

    @property
    def seqlen(self) -> int:
        return self.shape[1]

    def append(s1, s2: 'EndArray') -> 'EndArray':
        assert s1.endtype == s2.endtype
        n = np.vstack((s1, s2)).view(EndArray)
        n.endtype = s1.endtype
        return n

    def __repr__(self):
        return f"<endarray ({len(self)}): type {self.endtype}; {self.tolist()!r}>"

    def tolist(self) -> List[str]:  # noqa: UP006
        st = ["a", "c", "g", "t"]
        return ["".join([st[x] for x in y]) for y in self]

    strings = property(tolist)

endarray = EndArray


class Energetics(ABC):
    @abstractproperty
    def info(self) -> Dict[str, Any]:  # noqa: F821
        ...

    @abstractmethod
    def matching_uniform(self, seqs: Union[EndArray, np.ndarray]) -> np.ndarray:
        ...

    @abstractmethod
    def uniform(self, seqs1: Union[EndArray, np.ndarray], seqs2: Union[EndArray, np.ndarray]) -> np.ndarray:
        ...


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
