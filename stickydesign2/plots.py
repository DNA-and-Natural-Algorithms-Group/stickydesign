import seaborn as sns
import pandas as pd
import xarray
import numpy as np

def _pandas_data(pairarrays, namelists, models, modelnames):
    ecarrays = [np.array([model.gse_all(pa.ends,pa.pairs,forcemulti=True) for model in models]) 
               for pa in pairarrays]
    eearrays = [np.array([model.gse_all(pa.ends,pa.ends,forcemulti=True) for model in models]) 
               for pa in pairarrays]
    ccarrays = [np.array([model.gse_all(pa.pairs,pa.pairs,forcemulti=True) for model in models]) 
               for pa in pairarrays]
    
    diags = pd.concat( tuple(pd.DataFrame(x.diagonal(axis1=1,axis2=2).T, columns=modelnames, index=n) for x, n
                            in zip(ecarrays, namelists) ) )
    ecm = [x.copy() for x in ecarrays]
    eem = [x.copy() for x in eearrays]
    ccm = [x.copy() for x in ccarrays]
    for x,y,z in zip(ecm,eem,ccm):
        x[np.tile(np.eye(x.shape[1],dtype=bool),(x.shape[0],1,1))] = np.nan
        y[(slice(0,None),)+np.tril_indices(y.shape[1],k=-1)] = np.nan
        z[(slice(0,None),)+np.tril_indices(y.shape[1],k=-1)] = np.nan
    nondiagsec_panels = [xarray.DataArray(x, [modelnames,n,[a+'/' for a in n]]).to_dataset('dim_0')
                         for x, n in zip(ecm, namelists)]
    nondiagsee_panels = [xarray.DataArray(x, [modelnames,n,n]).to_dataset('dim_0') 
                         for x, n in zip(eem, namelists)]
    nondiagscc_panels = [xarray.DataArray(x, [modelnames,[a+'/' for a in n],[a+'/' for a in n]]).to_dataset('dim_0')
                         for x, n in zip(ccm, namelists)]
    nondiag = pd.concat( tuple(x.to_dataframe() for x in nondiagsec_panels+nondiagsee_panels+nondiagscc_panels) )
    eca_panels = [xarray.DataArray(x, [modelnames,n,[a+'/' for a in n]]).to_dataset('dim_0') 
                         for x, n in zip(ecarrays, namelists)]
    eea_panels = [xarray.DataArray(x, [modelnames,n,n]).to_dataset('dim_0')
                         for x, n in zip(eearrays, namelists)]
    cca_panels = [xarray.DataArray(x, [modelnames,[a+'/' for a in n],[a+'/' for a in n]]).to_dataset('dim_0') 
                         for x, n in zip(ccarrays, namelists)]
    allv = pd.concat( tuple(x.to_dataframe() for x in eca_panels+eea_panels+cca_panels) )
    return (diags, nondiag, allv)
