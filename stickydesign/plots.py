import numpy as np
from matplotlib import pylab

from .stickydesign import energy_array_uniform


def hist_multi(all_ends,
               all_energetics,
               energetics_names=None,
               title=None, **kwargs):
    
    pylab.figure(figsize=(10, 15))
    pylab.subplot(3, 1, 1)
    pylab.hist(
        [
            np.concatenate(tuple(en.matching_uniform(y) for y in all_ends))
            for en in all_energetics
        ],
        bins=50,
        label=energetics_names,
        **kwargs)
    if energetics_names:
        pylab.legend()
    pylab.xlabel("$ΔG_{se}$ (kcal/mol)")
    pylab.ylabel("# of interactions")
    pylab.title("Matching ends")
    pylab.subplot(3, 1, 2)
    pylab.hist(
        [
            np.concatenate(
                tuple(
                    np.ravel(energy_array_uniform(y, en)) for y in all_ends))
            for en in all_energetics
        ],
        bins=100,
        label=energetics_names, **kwargs)
    pylab.xlabel("$ΔG_{se}$ (kcal/mol)")
    pylab.ylabel("# of interactions")
    pylab.title("All ends")
    b = np.linspace(2.5, 9.0, 100)
    pylab.subplot(3, 1, 3)
    pylab.hist(
        [
            np.concatenate(
                tuple(
                    np.ravel(energy_array_uniform(y, en)) for y in all_ends))
            for en in all_energetics
        ],
        bins=b,
        label=energetics_names, **kwargs)
    pylab.xlim(2.5, 9.0)
    pylab.xlabel("$ΔG_{se}$ (kcal/mol)")
    pylab.ylabel("# of interactions")
    pylab.title("All ends (zoomed)")
    if title:
        pylab.suptitle(title)
    pylab.tight_layout(rect=[0, 0.3, 1, 0.97])
