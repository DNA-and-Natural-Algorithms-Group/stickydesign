import numpy as np
import itertools
import logging
from typing import Dict, Iterator, Optional, List, Tuple, Union, cast
from typing_extensions import TypeAlias
from collections.abc import Callable
from collections.abc import Sequence

from .endclasses import EndArray, lton, wc, Energetics, EndTypes
from .energetics_basic import EnergeticsBasic
from .energetics_daoe import EnergeticsDAOE

LOGGER = logging.getLogger(__name__)

EndChooserType: TypeAlias = Callable[[EndArray, EndArray, Energetics], np.ndarray]
SpaceFilterType: TypeAlias = Callable[[EndArray, Energetics], EndArray]
EndFilterType: TypeAlias = Callable[[EndArray, EndArray, EndArray, Energetics], EndArray]

__all__ = [
    'values_chunked', 'get_accept_set', 'find_end_set_uniform',
    'enhist', 'easyends', 'easy_space', 'spacefilter_standard',
    'endfilter_standard', 'endfilter_standard_advanced', 'energy_array_uniform',
    'endchooser_standard', 'endchooser_random'
]

def values_chunked(items: Sequence[Sequence[int]], endtype: str, chunk_dim: int = 10) -> Iterator[EndArray]:
    """
    Given a list of lists of acceptable numbers for each position in a row of
    an array, create every possible row, and return an iterator that returns
    chunks of every possible row up to chunk_dim, iterating dimensions higher
    than chunk_dim.  This probably doesn't need to be called directly, and may
    have a _ added in the future.

    Return this as an endarray, with set endtype. This can be easily emoved
    for use elsewhere.
    """
    ilengths = [len(x) for x in items]
    n = len(items)
    items = [np.array(x) for x in items]
    if n > chunk_dim:
        p = n - chunk_dim
        q = chunk_dim
        outer = itertools.product(*(items[0:p]))
    else:
        p = 0
        q = n

        def outer_iter():
            yield ()

        outer = outer_iter()

    chunk = np.zeros(
        [np.prod(ilengths[p:]), len(items)], dtype=int).view(EndArray)
    chunk.endtype = endtype
    chunk[:, p:] = np.indices(ilengths[p:]).reshape(q, -1).T
    for i in range(p, n):
        chunk[:, i] = items[i][chunk[:, i]]
    for seq in outer:
        chunk[:, :p] = seq
        yield chunk


def get_accept_set(endtype: EndTypes,
                   length: int,
                   interaction,
                   fdev,
                   maxendspurious,
                   spacefilter=None,
                   adjacents=('n', 'n'),
                   alphabet='n',
                   energetics=None):
    if not energetics:
        energetics = EnergeticsBasic()
    if not spacefilter:
        spacefilter = spacefilter_standard(interaction, interaction * fdev,
                                           maxendspurious)
    # Generate the template.
    if endtype == 'DT':
        template = [lton[adjacents[0]]] + [lton[alphabet.lower()]] \
                   * length + [lton[wc[adjacents[1]]]]
    elif endtype == 'TD':
        template = [lton[wc[adjacents[1]]]] + [lton[alphabet.lower()]] \
                   * length + [lton[adjacents[0]]]
    elif endtype == 'S':
        template = [lton[alphabet.lower()]]*length

    LOGGER.info("Length {}, type {}, adjacents {}, alphabet {}.".format(
        length, endtype, adjacents, alphabet))
    LOGGER.debug("Have template %s, endtype %s.", template, endtype)

    # Create the chunk iterator
    endchunk = values_chunked(template, endtype)

    # Use spacefilter to filter chunks down to usable sequences
    matcharrays = []
    totchunks = None
    totends = np.prod([len(x) for x in template])
    LOGGER.debug(
        "Have {} ends in total before any filtering.".format(totends))
    for chunknum, chunk in enumerate(endchunk, 1):
        matcharrays.append(spacefilter(chunk, energetics))
        if not totchunks:
            totchunks = totends // len(chunk)
        LOGGER.debug("Found {} filtered ends in chunk {} of {}.".format(
            len(matcharrays[-1]), chunknum, totchunks))
    LOGGER.debug("Done with spacefiltering.")
    availends = np.vstack(matcharrays).view(EndArray)
    availends.endtype = endtype
    return availends


def _make_avail(endtype: EndTypes,
                length: int,
                spacefilter: SpaceFilterType,
                endfilter: EndFilterType,
                endchooser: EndChooserType,
                energetics: Energetics,
                adjacents: Sequence[str] = ('n', 'n'),
                num: int=0,
                numtries: int=1,
                oldendfilter: Union[EndFilterType, None] = None,
                oldends: Union[EndArray, Sequence[str], None] = None,
                alphabet: str ='n') -> EndArray:
        # Generate the template.
    if endtype == 'DT':
        template = [lton[adjacents[0]]] + [lton[alphabet.lower()]] \
                   * length + [lton[wc[adjacents[1]]]]
    elif endtype == 'TD':
        template = [lton[wc[adjacents[1]]]] + [lton[alphabet.lower()]] \
                   * length + [lton[adjacents[0]]]
    elif endtype == 'S':
        template = [lton[alphabet.lower()]]*length
    
    LOGGER.info("Length {}, type {}, adjacents {}, alphabet {}.".format(
        length, endtype, adjacents, alphabet))
    LOGGER.debug("Have template %s, endtype %s", template, endtype)

    # Create the chunk iterator
    endchunk = values_chunked(template, endtype)

    # Use spacefilter to filter chunks down to usable sequences
    matcharrays = []
    totchunks = -1
    totends = np.prod([len(x) for x in template])
    LOGGER.debug(
        "Have {} ends in total before any filtering.".format(totends))
    for chunknum, chunk in enumerate(endchunk, 1):
        matcharrays.append(spacefilter(chunk, energetics))
        if not totchunks:
            totchunks = totends // len(chunk)
        LOGGER.debug("Found {} filtered ends in chunk {} of {}.".format(
            len(matcharrays[-1]), chunknum, totchunks))
    LOGGER.debug("Done with spacefiltering.")
    availends = np.vstack(matcharrays).view(EndArray)
    availends.endtype = endtype

    # Use endfilter to filter available sequences taking into account old
    # sequences.
    if oldends is not None and len(oldends) > 0:
        if oldendfilter:
            availends = oldendfilter(oldends, None, availends, energetics)
        else:
            availends = endfilter(oldends, None, availends, energetics)

    return availends


def find_end_set_uniform(endtype: EndTypes, 
                         length: int,
                         spacefilter: SpaceFilterType,
                         endfilter: EndFilterType,
                         endchooser: EndChooserType,
                         energetics: Energetics,
                         adjacents: Sequence[str] = ('n', 'n'),
                         num: int = 0,
                         numtries: int = 1,
                         oldendfilter=None,
                         oldends: Union[Sequence[str], EndArray] = (),
                         alphabet='n',
                         _presetavail=False) -> Union[EndArray, List[EndArray]]:  
    """
    Find a set of ends of uniform length and type satisfying uniform
    constraint functions (eg, constrant functions are the same for each
    end).

    This function is intended to be complicated and featureful. If you want
    something simpler, try easy_ends

    Parameters
    ----------
    endtype : str
      right now 'DT' for 3'-terminal ends, and 'TD' for
      5'-terminal ends,
    length : int
      length of ends, not including adjacent bases, if applicable.
    adjacents : list of str
      (defaults to ['n','n']): acceptable bases for adjacents
      (eg, ['n','n'] or ['c', 'c']) for the ends and their complements,
    num : int
      (defaults to 0): number of ends to find (0 keeps finding until
      available ends are exhausted)
    numtries : int
      (defaults to 1): if > 1, the function will return a list of
      sets of ends that all individually satisfy the constraints, so that
      the best one can be selected manually
    spacefilter: function
        a "spacefilter" function that takes endarrays and
        filters them down to ends that, not considering spurious
        interactions, are acceptable.
    endfilter: function
        an "endfilter" function that takes current ends in the
        set, available ends (filtered with current ends), and new ends added,
        and filters the available ends, considering interactions between ends
        (eg, spurious interactions).
    endchooser : function
        an "endchooser" function that takes current ends in the
        set and available ends, and returns a new end to add to the set.
    energetics : function
        an "energyfunctions" class that provides the energy
        functions for everything to use.
    oldends : endarray
        an endarray of old ends to consider as part of the set
    alphabet : str
        a single letter specifying what the alphabet for the ends
        should be (eg, four or three-letter code)
    oldendfilter : str
        a different "endfilter" function for use when filtering
        the available ends using interactions with old ends. This is normally
        not useful, but can be useful if you want, for example, to create a
        sets with higher cross-interactions between two subsets than within
        the two subsets.
    

    Returns
    -------
    endarray
      an endarray of generated ends, including provided old ends
    """ 
    if (len(oldends) > 0) and isinstance(oldends[0], str):
        oldends = EndArray(oldends, endtype)
    
    if isinstance(_presetavail, EndArray):
        startavail = _presetavail
    else:
        startavail = _make_avail(endtype,
                                 length,
                                 spacefilter,
                                 endfilter,
                                 endchooser,
                                 energetics,
                                 adjacents,
                                 num,
                                 numtries,
                                 oldendfilter,
                                 oldends,
                                 alphabet)
    endsets = []
    availends = startavail.copy()
    LOGGER.debug("Starting with %d ends.", len(availends))
    while len(endsets) < numtries:
        curends = oldends
        availends = startavail.copy()
        numends = 0
        while True:
            newend = EndArray(
                np.array([endchooser(curends, availends, energetics)]),
                endtype)
            LOGGER.debug(f"Chose end {newend!r}.")
            newend.endtype = endtype
            availends = endfilter(newend, curends, availends, energetics)
            LOGGER.debug("Done filtering.")
            if (curends is None) or (len(curends) == 0):
                curends = newend
            else:
                curends = curends.append(newend)
            numends += 1
            LOGGER.debug(f"Now have {numends} ends in set, and {len(availends)} ends available.")
            if len(availends) == 0:
                LOGGER.info("Found %d ends.", numends)
                break
            if numends >= num and num > 0:
                break
        endsets.append(curends)

    # Verification:
    # Note: this currently gives weird output that is not helpful when it fails.
    # But if this fails, you've done something very weird, most likely, because
    # this is just internal sanity checking.
    for endset in endsets:
        oldr = np.arange(0, len(oldends))
        newr = np.arange(len(oldends), len(endset))
        allr = np.arange(0, len(endset))
        # All new ends must satisfy old ends:
        if oldendfilter is None and len(oldends) > 0:
            assert np.asarray(
                endfilter(endset[oldr, :], None,
                          endset[newr, :], energetics) ==
                endset[newr, :]).all()
        elif len(oldends) > 0:
            assert np.asarray(
                oldendfilter(endset[oldr, :], None,
                             endset[newr, :], energetics) ==
                endset[newr, :]).all()
        # Each new end must allow all others
        for i in newr:
            if oldendfilter is None:
                assert np.asarray(
                    endfilter(endset[i, :][None, :], None,
                              endset, energetics) ==
                    endset[i != allr, :]).all()
            else:
                assert np.asarray(
                    oldendfilter(endset[i, :][None, :], None,
                                 endset[oldr, :], energetics) ==
                    endset[oldr, :]).all()
                assert np.asarray(
                    endfilter(endset[i, :][None, :], None,
                              endset[newr, :], energetics) ==
                    endset[newr[i != newr], :]).all()

    if len(endsets) > 1:
        return endsets
    elif _presetavail is None or isinstance(_presetavail,EndArray):
        return endsets[0], startavail
    else:
        return endsets[0]


def enhist(endtype: EndTypes,
           length: int,
           adjacents: Sequence[str] = ('n', 'n'),
           alphabet: str ='n',
           bins: Optional[np.ndarray] = None,
           energetics: Optional[Energetics] = None,
           plot: bool = False,
           color: str ='b') -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]: 
    if endtype == 'DT':
        template = [lton[adjacents[0]]] +\
                   [lton[alphabet.lower()]] * length + [lton[wc[adjacents[1]]]]
    elif endtype == 'TD':
        template = [lton[wc[adjacents[1]]]] +\
                   [lton[alphabet.lower()]] * length + [lton[adjacents[0]]]
    elif endtype == 'S':
        template = [lton[alphabet.lower()]]*length
    
    if not energetics:
        energetics = EnergeticsBasic()

    minbin = 0.8 * energetics.matching_uniform(
        EndArray([([0, 3] * length)[0:length + 2]], endtype))
    maxbin = 1.1 * energetics.matching_uniform(
        EndArray([([1, 2] * length)[0:length + 2]], endtype))

    if not bins:
        bins = np.arange(minbin, maxbin, 0.1)

    LOGGER.debug(f"Have template {template} and type {endtype}.")

    # Create the chunk iterator
    endchunk = values_chunked(template, endtype)

    hist = np.zeros(len(bins) - 1, dtype='int')
    totends = np.prod([len(x) for x in template])
    finishedends = 0
    info = {'min': np.inf, 'max': -np.inf, 'mean': 0}
    for chunk in endchunk:
        matchens = energetics.matching_uniform(chunk)
        hist += np.histogram(matchens, bins)[0]
        info['max'] = max(info['max'], np.amax(matchens))
        info['min'] = min(info['min'], np.amin(matchens))
        info['mean'] = (info['mean']*(finishedends)/(len(chunk)+finishedends)
                        + np.mean(matchens) * len(chunk) /
                        (len(chunk)+finishedends))
        finishedends += len(matchens)
        LOGGER.debug(f"Done with {finishedends}/{totends} ends.")

    x = (bins[:-1] + bins[1:]) / 2
    n = hist
    info['emean'] = np.sum(n * x, dtype='double') / np.sum(n, dtype='int64')
    info['estd'] = np.sqrt(
        np.sum(n * (
            x - info['emean'])**2, dtype='double') / np.sum(n, dtype='int64'))
    cs = np.cumsum(n)
    info['emedian'] = x[np.flatnonzero(cs >= cs[-1] / 2.0)[0]]

    if plot:
        import matplotlib.pyplot as plt

        plt.bar(
            bins[:-1],
            hist,
            width=(bins[1] - bins[0]),
            label=f"Type {endtype}, Length {length}, Adjs {adjacents}, Alph {alphabet}",
            color=color)
        plt.title(
            "Matching Energies of Ends of Type {}, Length {}, Adjs {}, Alph {}".
            format(endtype, length, adjacents, alphabet))
        plt.xlabel("Standard Free Energy (-kcal/mol)")
        plt.ylabel("Number of Ends")
        # plt.show()

    return (hist, bins, info)


def easyends(endtype: EndTypes,
             endlength: int,
             number: int = 0,
             interaction: Optional[float] =None,
             fdev: float=0.05,
             maxspurious: float=0.5,
             maxendspurious: Optional[float]=None,
             tries: int=1,
             oldends: Sequence[str]=(),
             adjs: Sequence[str]=('n', 'n'),
             energetics=None,
             alphabet: str='n',
             echoose: Optional[EndChooserType] = None,
             absolute: bool = False,
             _presetavail=False) -> EndArray:
    """
    Easyends is an attempt at creating an easy-to-use function for finding sets of ends.

    * endtype: specifies the type of end being considered. The system for
      classifying end types goes from 5' to 3', and consists of letters
      describing each side of the end. For example, an end that starts after a
      double-stranded region on the 5' side and ends at the end of the strand
      would be 'DT', while one that starts at the beginning of a strand on the
      5' side and ends in a double-stranded region would be 'TD'. 'T' stands
      for terminal, 'D' stands for double-stranded region, and 'S' stands for
      single-stranded region.
    * endlength: specifies the length of end being considered, not including
      adjacent bases.
    * number (optional): specifies the number of ends to find.  If zero or not
      provided, easyends tries to find as many ends as possible.
    * interaction (optional): a positive number corresponding to the desired
      standard free energy for hybridization of matching sticky ends. If not
      provided, easyends calculates an optimal value based on the sequence
      space.
    * fdev (default 0.05): the fractional deviation (above or below) of
      allowable matching energies.  maxspurious (default 0.5): the maximum
      spurious interaction, as a fraction of the matching interaction.
    * maxendspurious (default None): if provided, maxspurious is only used for
      spurious interactions between ends defined as ends, and ends defined as
      complements. Maxendspurious is then the maximum spurious interaction
      between ends and ends, and complements and complements. In a system
      where spurious interactions between ends and complements are more important
      than other spurious interactions, this can allow for better sets of ends.
    * tries (default 1): if > 1, easyends will return a list of sets of ends,
      all satisfying the constraints.
    * oldends (optional): a list of ends to be considered as already part of
      the set.
    * adjacents (default ['n','n']): allowable adjacent bases for ends and
      complements.
    * absolute (default False): fdev, maxspurious, and maxendspurious to be
      interpreted as absolute kcal/mol values rather than fractional values.
    * energetics (optional): an energetics class providing the energy
      calculation functions. You probably don't need to change this.
    * alphabet (default 'n'): The alphabet to use for ends, allowing
      for three-letter codes.
    """
    if energetics is None:
        if endtype in ['DT', 'TD']:  # noqa: SIM108
            energetics = EnergeticsDAOE()
        else:
            energetics = EnergeticsBasic()

    if (not interaction) or (interaction == 0):
        interaction = enhist(
            endtype,
            endlength,
            energetics=energetics,
            adjacents=adjs,
            alphabet=alphabet)[2]['emedian']
        LOGGER.info("Calculated optimal interaction energy is %s.", interaction)
    mult = interaction if not absolute else 1.0
        
    maxcompspurious = maxspurious * mult
    maxendspurious = maxspurious * mult if maxendspurious is None else maxendspurious * mult

    sfilt = spacefilter_standard(interaction, mult * fdev,
                                 maxendspurious)
    efilt = endfilter_standard_advanced(maxcompspurious, maxendspurious)
    if not echoose:
        echoose = endchooser_standard(interaction)

    return find_end_set_uniform(
        endtype,
        endlength,
        sfilt,
        efilt,
        echoose,
        energetics=energetics,
        numtries=tries,
        oldends=oldends,
        adjacents=adjs,
        num=number,
        alphabet=alphabet,
        _presetavail=_presetavail)


def easy_space(endtype: EndTypes,
               endlength: int,
               interaction: Optional[float] = None,
               fdev:float = 0.05,
               maxspurious: float = 0.5,
               maxendspurious: Optional[float] = None,
               tries: int = 1,
               oldends: Sequence[str] = (),
               adjs: Sequence[str] = ('n', 'n'),
               energetics: Optional[Energetics] = None,
               alphabet: str = 'n',
               echoose: Optional[EndChooserType] = None):
    length = endlength
    if not energetics:
        efunc = EnergeticsBasic()
        energetics = efunc
    else:
        efunc = energetics
    if (not interaction) or (interaction == 0):
        interaction = cast(float, enhist(
            endtype,
            endlength,
            energetics=efunc,
            adjacents=adjs,
            alphabet=alphabet)[2]['emedian'])
        LOGGER.info("Calculated optimal interaction energy is %g.",
            interaction)
    maxcompspurious = maxspurious * interaction
    if not maxendspurious:
        maxendspurious = maxspurious * interaction
    else:
        maxendspurious = maxendspurious * interaction

    sfilt = spacefilter_standard(interaction, interaction * fdev,
                                 maxendspurious)
    spacefilter = sfilt

    if not echoose:
        echoose = endchooser_standard(interaction)

    adjacents = adjs

    if endtype == 'DT':
        template = [lton[adjacents[0]]] + [lton[alphabet.lower()]] \
                   * length + [lton[wc[adjacents[1]]]]
    elif endtype == 'TD':
        template = [lton[wc[adjacents[1]]]] + [lton[alphabet.lower()]] \
                   * length + [lton[adjacents[0]]]

    # Create the chunk iterator
    endchunk = values_chunked(template, endtype)

    # Use spacefilter to filter chunks down to usable sequences
    matcharrays = []
    totchunks = None
    totends = np.prod([len(x) for x in template])
    LOGGER.info(
        "Have {} ends in total before any filtering.".format(totends))
    for chunknum, chunk in enumerate(endchunk, 1):
        matcharrays.append(spacefilter(chunk, energetics))
        if not totchunks:
            totchunks = totends // len(chunk)
        LOGGER.debug("Found {0} filtered ends in chunk {1} of {2}.".format(
            len(matcharrays[-1]), chunknum, totchunks))
    LOGGER.debug("Done with spacefiltering.")
    availends = np.vstack(matcharrays).view(EndArray)
    availends.endtype = endtype

    availendsr = cast(EndArray, np.repeat(availends, len(availends), axis=0))
    availendst = cast(EndArray, np.tile(availends, (len(availends), 1)))

    vals_ee = energetics.uniform(availendsr.ends, availendst.ends)
    vals_ec = energetics.uniform(availendsr.ends, availendst.comps)
    vals_ce = energetics.uniform(availendsr.comps, availendst.ends)
    vals_cc = energetics.uniform(availendsr.comps, availendst.comps)

    vals_tf = ((vals_ee < maxendspurious) & (vals_cc < maxendspurious) &
               (vals_ec < maxcompspurious) & (vals_ce < maxcompspurious))
    zipendsnf = list(zip(availendsr.tolist(), availendst.tolist(), strict=True))
    zipends = [zipendsnf[x] for x in np.flatnonzero(vals_tf)]

    return zipends

def spacefilter_standard(desint: float, dev: float, maxself: float) -> SpaceFilterType:
    """
    A spacefilter function: filters to ends that have a end-complement
    interaction of between desint-dev and desint+dev, and a self-interaction
    (end-end or comp-comp) of less than maxself.
    """

    def spacefilter(fullends: EndArray, energetics: Energetics) -> EndArray:
        matchenergies = energetics.matching_uniform(fullends)
        g4 = np.zeros(fullends.shape[0])
        for w in range(fullends.shape[1] - 3):
            g4 += (np.sum(
                np.array(fullends[:, w:(w + 4)] == [2, 2, 2, 2]), axis=1) == 4)
            g4 += (np.sum(
                np.array(fullends[:, w:(w + 4)] == [1, 1, 1, 1]), axis=1) == 4)
        i = np.flatnonzero((matchenergies < desint + dev) &
                           (matchenergies > desint - dev) & (g4 == 0))
        matchenergies = matchenergies[i]
        fullends = cast(EndArray, fullends[i])
        selfselfenergies = energetics.uniform(fullends.ends, fullends.ends)
        compcompenergies = energetics.uniform(fullends.comps, fullends.comps)
        i = np.flatnonzero((selfselfenergies < maxself) & (compcompenergies <
                                                           maxself))
        return cast(EndArray, fullends[i])

    return spacefilter


def endfilter_standard(maxspurious: float) -> EndFilterType:
    """
    An endfilter function: filters out ends that have any (end-end, end-comp,
    comp-end, comp-comp) interactions with new ends above maxspurious.
    """

    def endfilter(newends: EndArray, currentends: EndArray, availends: EndArray, energetics: Energetics) -> EndArray:
        endendspurious = energetics.uniform(
            np.repeat(newends.ends, len(availends), 0),
            np.tile(availends.ends, (len(newends), 1))).reshape(
                len(availends), len(newends), order='F')
        endcompspurious = energetics.uniform(
            np.repeat(newends.ends, len(availends), 0),
            np.tile(availends.comps, (len(newends), 1))).reshape(
                len(availends), len(newends), order='F')
        compendspurious = energetics.uniform(
            np.repeat(newends.comps, len(availends), 0),
            np.tile(availends.ends, (len(newends), 1))).reshape(
                len(availends), len(newends), order='F')
        compcompspurious = energetics.uniform(
            np.repeat(newends.comps, len(availends), 0),
            np.tile(availends.comps, (len(newends), 1))).reshape(
                len(availends), len(newends), order='F')
        highspurious = np.amax(
            np.hstack((endendspurious, compendspurious, endcompspurious,
                       compcompspurious)), 1)

        return availends[highspurious < maxspurious]

    return endfilter


def endfilter_standard_advanced(maxcompspurious: float, maxendspurious: float) -> EndFilterType:
    """
    An endfilter function: filters out ends that have end-comp or comp-end
    interactions above maxcompspurious, and end-end or comp-comp interactions
    above maxendspurious.
    """

    def endfilter(newends: EndArray, currentends: EndArray, availends: EndArray, energetics: Energetics) -> EndArray:
        endendspurious = energetics.uniform(
            np.repeat(newends.ends, len(availends), 0),
            np.tile(availends.ends, (len(newends), 1))).reshape(
                len(availends), len(newends), order='F')
        endcompspurious = energetics.uniform(
            np.repeat(newends.ends, len(availends), 0),
            np.tile(availends.comps, (len(newends), 1))).reshape(
                len(availends), len(newends), order='F')
        compendspurious = energetics.uniform(
            np.repeat(newends.comps, len(availends), 0),
            np.tile(availends.ends, (len(newends), 1))).reshape(
                len(availends), len(newends), order='F')
        compcompspurious = energetics.uniform(
            np.repeat(newends.comps, len(availends), 0),
            np.tile(availends.comps, (len(newends), 1))).reshape(
                len(availends), len(newends), order='F')
        highendspurious = np.amax(
            np.hstack((endendspurious, compcompspurious)), 1)
        highcompspurious = np.amax(
            np.hstack((compendspurious, endcompspurious)), 1)

        return availends[(highendspurious < maxendspurious)
                         & (highcompspurious < maxcompspurious)]

    return endfilter


def energy_array_uniform(seqs: EndArray, energetics: Energetics) -> np.ndarray:
    """Given an endarray and a set of sequences, return an array of the interactions between them, including their complements."""
    seqs = seqs.ends.append(seqs.comps)
    return energetics.uniform(
        np.repeat(seqs, seqs.shape[0], 0), np.tile(
            seqs, (seqs.shape[0], 1))).reshape((seqs.shape[0], seqs.shape[0]))


def endchooser_standard(desint: float, wiggle: float = 0.0) -> EndChooserType:
    """Return a random end with end-comp energy closest to desint, subject to some wiggle."""
    rng = np.random.default_rng()

    def endchooser(currentends: EndArray, availends: EndArray, energetics: Energetics) -> EndArray:
        ddiff = np.abs(energetics.matching_uniform(availends) - desint)
        choices = np.flatnonzero(ddiff <= np.amin(ddiff)+wiggle)
        newend = availends[choices[rng.integers(0, len(choices))]]
        return newend

    return endchooser


def endchooser_random() -> EndChooserType:
    """Return a random end with end-comp energy closest to desint."""
    rng = np.random.default_rng()

    def endchooser(currentends: EndArray, availends: EndArray, energetics: Energetics) -> EndArray:
        newend = availends[rng.integers(0, len(availends))]
        return newend

    return endchooser


