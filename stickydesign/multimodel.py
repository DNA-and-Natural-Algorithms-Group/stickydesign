import numpy as np
from math import ceil


def endchooser(all_energetics,
               target_vals=None,
               init_wigglefraction=1,
               next_wigglefraction=0.1):
    """
    An endchooser generator that chooses ends while trying to optimize for
    multiple energy models simultaneously.

    Arguments:

    `all_energetics`: a list of energetics models to optimize for.

    `target_vals`: if provided, a list or numpy array of target energy values
    for each model.  This is primarily useful if you have already generated
    sticky ends, and want to choose ends that match (eg, if you chose DT ends
    and now want to choose TD ones).
    """
    
    def endchooser(currentends, availends, energetics):
        nonlocal target_vals
        if len(currentends) == 0 and not target_vals:
            # Starting out, we need to choose an initial end, and use that
            # to choose our target values.
            dev = np.std(
                [en.matching_uniform(availends) for en in all_energetics],
                axis=0)
            choices = np.argsort(dev)
            choice = choices[np.random.randint(
                0,
                max(1, ceil(init_wigglefraction * len(availends)))
                )]
            return availends[choice]
        else:
            if not target_vals:  # NOQA
                target_vals = [en.matching_uniform(currentends[0:1])
                               for en in all_energetics]
            dev = np.sqrt(
                np.sum(
                    np.array([en.matching_uniform(availends) - target_val
                              for en, target_val
                              in zip(all_energetics, target_vals)])**2,
                    axis=0
                    )
                )
            choices = np.argsort(dev)
            choice = choices[np.random.randint(
                0,
                max(1, ceil(len(choices) * next_wigglefraction)))]
            return availends[choice]
    return endchooser


def deviation_score(all_ends, all_energetics):
    return np.sqrt(
        np.sum(
            np.var([np.concatenate(tuple(en.matching_uniform(ends)
                                         for ends in all_ends))
                    for en in all_energetics],
                   axis=1)
            ))
