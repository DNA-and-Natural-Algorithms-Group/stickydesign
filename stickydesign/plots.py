import numpy as np


from .stickydesign import energy_array_uniform


def hist_multi(all_ends,
               all_energetics,
               energetics_names=None,
               title="", **kwargs):
    from matplotlib import pylab
    fig = pylab.figure(figsize=(10, 15))
    a, b, c = fig.subplots(3, 1)
    a.hist(
        [
            np.concatenate(tuple(en.matching_uniform(y) for y in all_ends))
            for en in all_energetics
        ],
        bins=50,
        label=energetics_names,
        **kwargs)
    if energetics_names:
        a.legend()
    a.set_xlabel("$ΔG_{se}$ (kcal/mol)")
    a.set_ylabel("# of interactions")
    a.set_title("Matching ends")

    b.hist(
        [
            np.concatenate(
                tuple(
                    np.ravel(energy_array_uniform(y, en)) for y in all_ends))
            for en in all_energetics
        ],
        bins=100,
        label=energetics_names, **kwargs)
    b.set_xlabel("$ΔG_{se}$ (kcal/mol)")
    b.set_ylabel("# of interactions")
    b.set_title("All ends")
    bins = np.linspace(2.5, 9.0, 100)

    c.hist(
        [
            np.concatenate(
                tuple(
                    np.ravel(energy_array_uniform(y, en)) for y in all_ends))
            for en in all_energetics
        ],
        bins=bins,
        label=energetics_names, **kwargs)
    c.set_xlim(2.5, 9.0)
    c.set_xlabel("$ΔG_{se}$ (kcal/mol)")
    c.set_ylabel("# of interactions")
    c.set_title("All ends (zoomed)")
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.3, 1, 0.97])
    return fig

def box_multi(all_ends,
              all_energetics,
              energetics_names=None,
              title="", **kwargs):
    from matplotlib import pylab
    fig = pylab.figure(figsize=(10, 15))
    a = fig.subplots(1, 1)
    a.boxplot(
        [
            np.concatenate(tuple(en.matching_uniform(y) for y in all_ends))
            for en in all_energetics
        ],
        labels=energetics_names, sym='r.',
        **kwargs)
    a.set_ylabel("$ΔG_{se}$ (kcal/mol)")

    v = []
    for en in all_energetics:
        l = []
        for y in all_ends:
            x = np.ma.masked_array(energy_array_uniform(y, en))
            i = np.arange(0,len(x)//2)
            x[i,len(x)//2+i]=np.ma.masked
            x[np.tril_indices_from(x,-1)]=np.ma.masked
            l.append(x.compressed())
        v.append(np.concatenate(tuple(l)))
    
    a.boxplot(v, sym='bP', labels=energetics_names, **kwargs)

    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.3, 1, 0.97])
    return fig


def heatmap(ends, energetics, title="", **kwargs):
    from matplotlib import pylab
    fig = pylab.figure()

    heat = energy_array_uniform(ends, energetics)

    pylab.imshow(heat)
    pylab.colorbar()
    pylab.title(title)

    return fig


def _multi_data_pandas(ends, all_energetics, energetics_names=None):
    """From a list of endarrays, make two Pandas DataFrames: one of
    matching energies, the other of spurious/mismatch energies.

    Returns
    -------

    match : pandas.DataFrame
        matching energies

    mismatch: pandas.DataFrame
        mismatch energies
    """
    import pandas as pd
    match = np.array([np.concatenate(
        [e.matching_uniform(x) for x in ends]) for e in all_energetics]).T
    match = pd.DataFrame(match, columns=energetics_names)
    
    v = []
    for en in all_energetics:
        l = []
        for y in ends:
            x = np.ma.masked_array(energy_array_uniform(y, en))
            i = np.arange(0, len(x)//2)
            x[i, len(x)//2+i] = np.ma.masked
            x[np.tril_indices_from(x, -1)] = np.ma.masked
            l.append(x.compressed())
        v.append(np.concatenate(tuple(l)))
    return match, pd.DataFrame(np.array(v).T, columns=energetics_names)
