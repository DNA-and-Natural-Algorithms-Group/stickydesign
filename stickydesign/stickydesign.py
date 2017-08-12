from __future__ import division
import numpy as np
import itertools
import logging
from math import ceil

from .endclasses import endarray, lton, wc
from .energetics import energetics_santalucia


def values_chunked(items, endtype, chunk_dim=10):
    """
    Given a list of lists of acceptable numbers for each position in a row of
    an array, create every possible row, and return an iterator that returns
    chunks of every possible row up to chunk_dim, iterating dimensions higher
    than chunk_dim.

    Return this as an endarray, with set endtype. This can be easily removed
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
        [np.prod(ilengths[p:]), len(items)], dtype=int).view(endarray)
    chunk.endtype = endtype
    chunk[:, p:] = np.indices(ilengths[p:]).reshape(q, -1).T
    for i in range(p, n):
        chunk[:, i] = items[i][chunk[:, i]]
    for seq in outer:
        chunk[:, :p] = seq
        yield chunk


def get_accept_set(endtype,
                   length,
                   interaction,
                   fdev,
                   maxendspurious,
                   spacefilter=None,
                   adjacents=['n', 'n'],
                   alphabet='n',
                   energetics=None):
    if not energetics:
        energetics = energetics_santalucia(mismatchtype='max')
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

    logging.info("Length {0}, type {1}, adjacents {2}, alphabet {3}.".format(
        length, endtype, adjacents, alphabet))
    logging.debug("Have template {0}.".format(template, endtype))

    # Create the chunk iterator
    endchunk = values_chunked(template, endtype)

    # Use spacefilter to filter chunks down to usable sequences
    matcharrays = []
    chunknum = 0
    totchunks = None
    totends = np.product([len(x) for x in template])
    logging.debug(
        "Have {0} ends in total before any filtering.".format(totends))
    for chunk in endchunk:
        matcharrays.append(spacefilter(chunk, energetics))
        if not totchunks:
            totchunks = totends // len(chunk)
        chunknum += 1
        logging.debug("Found {0} filtered ends in chunk {1} of {2}.".format(
            len(matcharrays[-1]), chunknum, totchunks))
    logging.debug("Done with spacefiltering.")
    availends = np.vstack(matcharrays).view(endarray)
    availends.endtype = endtype
    return availends


def find_end_set_uniform(endtype,
                         length,
                         spacefilter,
                         endfilter,
                         endchooser,
                         energetics,
                         adjacents=['n', 'n'],
                         num=0,
                         numtries=1,
                         oldendfilter=None,
                         oldends=[],
                         alphabet='n'):
    """
        Find a set of ends of uniform length and type satisfying uniform
        constraint functions (eg, constrant functions are the same for each
        end).

        * endtype: right now 'DT' for 3'-terminal ends, and 'TD' for
          5'-terminal ends,
        * length: length of ends, not including adjacents.
        * adjacents (defaults to ['n','n']): acceptable bases for adjacents
          (eg, ['n','n'] or ['c', 'c']) for the ends and their complements,
        * num (defaults to 0): number of ends to find (0 keeps finding until
          available ends are exhausted)
        * numtries (defaults to 1): if > 1, the function will return a list of
          sets of ends that all individually satisfy the constraints, so that
          the best one can be selected manually
        * spacefilter: a "spacefilter" function that takes endarrays and
          filters them down to ends that, not considering spurious
          interactions, are acceptable.
        * endfilter: an "endfilter" function that takes current ends in the
          set, available ends (filtered with current ends), and new ends added,
          and filters the available ends, considering interactions between ends
          (eg, spurious interactions).
        * endchooser: an "endchooser" function that takes current ends in the
          set and available ends, and returns a new end to add to the set.
        * energetics: an "energyfunctions" class that provides the energy
          functions for everything to use.
        * oldends: an endarray of old ends to consider as part of the set
        * alphabet: a single letter specifying what the alphabet for the ends
          should be (eg, four or three-letter code)
        * oldendfilter: a different "endfilter" function for use when filtering
          the available ends using interactions with old ends. This is normally
          not useful, but can be useful if you want, for example, to create a
          sets with higher cross-interactions between two subsets than within
          the two subsets.

        This function is intended to be complicated and featureful. If you want
        something simpler, try easy_ends
    """
    # Generate the template.
    if endtype == 'DT':
        template = [lton[adjacents[0]]] + [lton[alphabet.lower()]] \
                   * length + [lton[wc[adjacents[1]]]]
    elif endtype == 'TD':
        template = [lton[wc[adjacents[1]]]] + [lton[alphabet.lower()]] \
                   * length + [lton[adjacents[0]]]

    logging.info("Length {0}, type {1}, adjacents {2}, alphabet {3}.".format(
        length, endtype, adjacents, alphabet))
    logging.debug("Have template {0}.".format(template, endtype))

    # Create the chunk iterator
    endchunk = values_chunked(template, endtype)

    # Use spacefilter to filter chunks down to usable sequences
    matcharrays = []
    chunknum = 0
    totchunks = None
    totends = np.product([len(x) for x in template])
    logging.debug(
        "Have {0} ends in total before any filtering.".format(totends))
    for chunk in endchunk:
        matcharrays.append(spacefilter(chunk, energetics))
        if not totchunks:
            totchunks = totends // len(chunk)
        chunknum += 1
        logging.debug("Found {0} filtered ends in chunk {1} of {2}.".format(
            len(matcharrays[-1]), chunknum, totchunks))
    logging.debug("Done with spacefiltering.")
    availends = np.vstack(matcharrays).view(endarray)
    availends.endtype = endtype

    # Use endfilter to filter available sequences taking into account old
    # sequences.
    if len(oldends) > 0:
        if type(oldends[0]) is str:
            oldends = endarray(oldends, endtype)
        if oldendfilter:
            availends = oldendfilter(oldends, None, availends, energetics)
        else:
            availends = endfilter(oldends, None, availends, energetics)

    startavail = availends
    endsets = []
    logging.debug("Starting with {0} ends.".format(len(availends)))
    while len(endsets) < numtries:
        curends = oldends
        availends = startavail
        numends = 0
        while True:
            newend = endarray(
                np.array([endchooser(curends, availends, energetics)]),
                endtype)
            logging.debug("Chose end {0}.".format(repr(newend)))
            newend.endtype = endtype
            availends = endfilter(newend, curends, availends, energetics)
            logging.debug("Done filtering.")
            if curends is None:
                curends = newend
            elif len(curends) == 0:
                curends = newend
            else:
                curends = curends.append(newend)
            numends += 1
            logging.debug("Now have {0} ends in set, and {1} ends available.".
                          format(numends, len(availends)))
            if len(availends) == 0:
                logging.info("Found {0} ends.".format(numends))
                break
            if numends >= num and num > 0:
                break
        endsets.append(curends)

    if len(endsets) > 1:
        return endsets
    else:
        return endsets[0]
    # TODO: multiple sets generated, decide which is best.


def enhist(endtype,
           length,
           adjacents=['n', 'n'],
           alphabet='n',
           bins=None,
           energetics=None,
           plot=False,
           color='b'):
    if endtype == 'DT':
        template = [lton[adjacents[0]]] +\
                   [lton[alphabet.lower()]] * length + [lton[wc[adjacents[1]]]]
    elif endtype == 'TD':
        template = [lton[wc[adjacents[1]]]] +\
                   [lton[alphabet.lower()]] * length + [lton[adjacents[0]]]

    if not energetics:
        energetics = energetics_santalucia(mismatchtype='max')

    minbin = 0.8 * energetics.matching_uniform(
        endarray([([0, 3] * length)[0:length + 2]], endtype))
    maxbin = 1.1 * energetics.matching_uniform(
        endarray([([1, 2] * length)[0:length + 2]], endtype))

    if not bins:
        bins = np.arange(minbin, maxbin, 0.1)

    logging.debug("Have template {0} and type {1}.".format(template, endtype))

    # Create the chunk iterator
    endchunk = values_chunked(template, endtype)

    hist = np.zeros(len(bins) - 1, dtype='int')
    totends = np.product([len(x) for x in template])
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
        logging.debug("Done with {0}/{1} ends.".format(finishedends, totends))

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
            label="Type {3}, Length {0}, Adjs {1}, Alph {2}".format(
                length, adjacents, alphabet, endtype),
            color=color)
        plt.title(
            "Matching Energies of Ends of Type {3}, Length {0}, Adjs {1}, Alph {2}".
            format(length, adjacents, alphabet, endtype))
        plt.xlabel("Standard Free Energy (-kcal/mol)")
        plt.ylabel("Number of Ends")
        # plt.show()

    return (hist, bins, info)


def easyends(endtype,
             endlength,
             number=0,
             interaction=None,
             fdev=0.05,
             maxspurious=0.5,
             maxendspurious=None,
             tries=1,
             oldends=[],
             adjs=['n', 'n'],
             energetics=None,
             alphabet='n',
             echoose=None):
    """
    Easyends is an attempt at creating an easy-to-use function for finding sets
    of ends.

    * endtype: specifies the type of end being considered. The system for
      classifying end types goes from 5' to 3', and consists of letters
      describing each side of the end. For example, an end that starts after a
      double-stranded region on the 5' side and ends at the end of the strand
      would be 'DT', while one that starts at the beginning of a strand on the
      5' side and ends in a double-stranded region would be 'TD'. 'T' stands
      for terminal, 'D' stands for double-stranded region, and 'S' stands for
      single-stranded region. 'S', however, is not currently supported.
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
    * energetics (optional): an energetics class providing the energy
      calculation functions. You probably don't need to change this.
    * alphabet (default 'n'): The alphabet to use for ends, allowing
      for three-letter codes.
    """

    if not energetics:
        efunc = energetics_santalucia(mismatchtype='max')
    else:
        efunc = energetics
    if (not interaction) or (interaction == 0):
        interaction = enhist(
            endtype,
            endlength,
            energetics=efunc,
            adjacents=adjs,
            alphabet=alphabet)[2]['emedian']
        logging.info("Calculated optimal interaction energy is {0}.".format(
            interaction))
    maxcompspurious = maxspurious * interaction
    if not maxendspurious:
        maxendspurious = maxspurious * interaction
    else:
        maxendspurious = maxendspurious * interaction

    sfilt = spacefilter_standard(interaction, interaction * fdev,
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
        energetics=efunc,
        numtries=tries,
        oldends=oldends,
        adjacents=adjs,
        num=number,
        alphabet=alphabet)


def easy_space(endtype,
               endlength,
               interaction=None,
               fdev=0.05,
               maxspurious=0.5,
               maxendspurious=None,
               tries=1,
               oldends=[],
               adjs=['n', 'n'],
               energetics=None,
               alphabet='n',
               echoose=None):
    length = endlength
    if not energetics:
        efunc = energetics_santalucia(mismatchtype='max')
        energetics = efunc
    else:
        efunc = energetics
    if (not interaction) or (interaction == 0):
        interaction = enhist(
            endtype,
            endlength,
            energetics=efunc,
            adjacents=adjs,
            alphabet=alphabet)[2]['emedian']
        logging.info("Calculated optimal interaction energy is {0}.".format(
            interaction))
    maxcompspurious = maxspurious * interaction
    if not maxendspurious:
        maxendspurious = maxspurious * interaction
    else:
        maxendspurious = maxendspurious * interaction

    sfilt = spacefilter_standard(interaction, interaction * fdev,
                                 maxendspurious)
    spacefilter = sfilt
    efilt = endfilter_standard_advanced(maxcompspurious, maxendspurious)
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
    chunknum = 0
    totchunks = None
    totends = np.product([len(x) for x in template])
    logging.info(
        "Have {0} ends in total before any filtering.".format(totends))
    for chunk in endchunk:
        matcharrays.append(spacefilter(chunk, energetics))
        if not totchunks:
            totchunks = totends // len(chunk)
        chunknum += 1
        logging.debug("Found {0} filtered ends in chunk {1} of {2}.".format(
            len(matcharrays[-1]), chunknum, totchunks))
    logging.debug("Done with spacefiltering.")
    availends = np.vstack(matcharrays).view(endarray)
    availends.endtype = endtype

    availendsr = np.repeat(availends, len(availends), axis=0)
    availendst = np.tile(availends, (len(availends), 1))

    vals_ee = energetics.uniform(availendsr.ends, availendst.ends)
    vals_ec = energetics.uniform(availendsr.ends, availendst.comps)
    vals_ce = energetics.uniform(availendsr.comps, availendst.ends)
    vals_cc = energetics.uniform(availendsr.comps, availendst.comps)

    vals_tf = ((vals_ee < maxendspurious) & (vals_cc < maxendspurious) &
               (vals_ec < maxcompspurious) & (vals_ce < maxcompspurious))
    zipendsnf = zip(availendsr.tolist(), availendst.tolist())
    zipends = [zipendsnf[x] for x in np.flatnonzero(vals_tf)]

    return zipends


def spacefilter_standard(desint, dev, maxself):
    """
    A spacefilter function: filters to ends that have a end-complement
    interaction of between desint-dev and desint+dev, and a self-interaction
    (end-end or comp-comp) of less than maxself.
    """

    def spacefilter(fullends, energetics):
        matchenergies = energetics.matching_uniform(fullends)
        g4 = np.zeros(fullends.shape[0])
        for w in range(0, (fullends.shape[1] - 3)):
            g4 += (np.sum(
                np.array(fullends[:, w:(w + 4)] == [2, 2, 2, 2]), axis=1) == 4)
            g4 += (np.sum(
                np.array(fullends[:, w:(w + 4)] == [1, 1, 1, 1]), axis=1) == 4)
        i = np.flatnonzero((matchenergies < desint + dev) &
                           (matchenergies > desint - dev) & (g4 == 0))
        matchenergies = matchenergies[i]
        fullends = fullends[i]
        selfselfenergies = energetics.uniform(fullends.ends, fullends.ends)
        compcompenergies = energetics.uniform(fullends.comps, fullends.comps)
        i = np.flatnonzero((selfselfenergies < maxself) & (compcompenergies <
                                                           maxself))
        return fullends[i]

    return spacefilter


def endfilter_standard(maxspurious):
    """
    An endfilter function: filters out ends that have any (end-end, end-comp,
    comp-end, comp-comp) interactions with new ends above maxspurious.
    """

    def endfilter(newends, currentends, availends, energetics):
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


def endfilter_standard_advanced(maxcompspurious, maxendspurious):
    """
    An endfilter function: filters out ends that have end-comp or comp-end
    interactions above maxcompspurious, and end-end or comp-comp interactions
    above maxendspurious.
    """

    def endfilter(newends, currentends, availends, energetics):
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


def energy_array_uniform(seqs, energetics):
    """
    Given an endarray and a set of sequences, return an array of the
    interactions between them, including their complements.
    """
    seqs = seqs.ends.append(seqs.comps)
    return energetics.uniform(
        np.repeat(seqs, seqs.shape[0], 0), np.tile(
            seqs, (seqs.shape[0], 1))).reshape((seqs.shape[0], seqs.shape[0]))


def endchooser_standard(desint, wiggle=0.0):
    """
    An endchooser function: return a random end with end-comp energy closest to
    desint.
    """

    def endchooser(currentends, availends, energetics):
        ddiff = np.abs(energetics.matching_uniform(availends) - desint)
        choices = np.flatnonzero(ddiff <= np.amin(ddiff)+wiggle)
        newend = availends[choices[np.random.randint(0, len(choices))]]
        return newend

    return endchooser


def endchooser_random():
    """
    An endchooser function: return a random end with end-comp energy closest to
    desint.
    """

    def endchooser(currentends, availends, energetics):
        newend = availends[np.random.randint(0, len(availends))]
        return newend

    return endchooser



