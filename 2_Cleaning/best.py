import vaex
import numpy as np
from glob import glob
from time import time
from os.path import join, abspath
from os import pardir, mkdir
import sys
from matplotlib import pyplot as plt

# import utils
util_dir = abspath(pardir)
sys.path.insert(0, util_dir)

from utils import progressbar

root_data_dir = abspath(join(pardir, "Data"))

name = "Best-Pars"
data_dir = join(root_data_dir, name)
try:
    mkdir(data_dir)
    print(f"Creating {name} dir in Data dir")
except FileExistsError:
    print("Directory already exist. Good to go!")
    
name = "rave-galah-lamost-apogee"
combine_data_dir = join(root_data_dir, 'Combine', name)

files = glob(join(combine_data_dir, "*.hdf5"))
files.sort()

# cols initiation
rv_cols = [
    {
        'value': 'rv_gaia', 
        'error': 'e_rv_gaia', 
        'cat': 'gaia'
    }, 
    {
        'value': 'rv_rave', 
        'error': 'e_rv_rave', 
        'cat': 'rave'
    },
    {
        'value': 'rv_apogee', 
        'error': 'e_rv_apogee', 
        'cat': 'apogee'
    },
    {
        'value': 'rv_lamost', 
        'error': 'e_rv_lamost', 
        'cat': 'lamost'
    },
    {
        'value': 'rv_galah', 
        'error': 'e_rv_galah', 
        'cat': 'galah'
    }
]

teff_cols = [
    {
        'value': 'teff_gspphot', 
        'error': {
            'upper': 'teff_gspphot_upper',
            'lower': 'teff_gspphot_lower'
        }, 
        'cat': 'gspphot'
    }, 
    {
        'value': 'teff_rave', 
        'error': np.nan, 
        'cat': 'rave'
    },
    {
        'value': 'teff_apogee', 
        'error': 'e_teff_apogee', 
        'cat': 'apogee'
    },
    {
        'value': 'teff_lamost', 
        'error': 'e_teff_lamost', 
        'cat': 'lamost'
    },
    {
        'value': 'teff_galah', 
        'error': 'e_teff_galah', 
        'cat': 'galah'
    }
]

logg_cols = [
    {
        'value': 'logg_gspphot', 
        'error': {
            'upper': 'logg_gspphot_upper',
            'lower': 'logg_gspphot_lower'
        }, 
        'cat': 'gspphot'
    }, 
    {
        'value': 'logg_rave', 
        'error': np.nan, 
        'cat': 'rave'
    },
    {
        'value': 'logg_apogee', 
        'error': 'e_logg_apogee', 
        'cat': 'apogee'
    },
    {
        'value': 'logg_lamost', 
        'error': 'e_logg_lamost', 
        'cat': 'lamost'
    },
    {
        'value': 'logg_galah', 
        'error': 'e_logg_galah', 
        'cat': 'galah'
    }
]

mh_cols = [
    {
        'value': 'mh_gspphot', 
        'error': {
            'upper': 'mh_gspphot_upper',
            'lower': 'mh_gspphot_lower'
        }, 
        'cat': 'gspphot'
    },
    {
        'value': 'mh_gspspec', 
        'error': {
            'upper': 'mh_gspspec_upper',
            'lower': 'mh_gspspec_lower'
        }, 
        'cat': 'gspspec'
    },
    {
        'value': 'mh_rave', 
        'error': np.nan, 
        'cat': 'rave'
    },
    {
        'value': 'mh_apogee', 
        'error': 'e_mh_apogee', 
        'cat': 'apogee'
    }
]

alphafe_cols = [
    {
        'value': 'alphafe_gspspec', 
        'error': {
            'upper': 'alphafe_gspspec_upper',
            'lower': 'alphafe_gspspec_lower'
        }, 
        'cat': 'gspspec'
    },
    {
        'value': 'alphafe_rave', 
        'error': np.nan, 
        'cat': 'rave'
    },
    {
        'value': 'alphafe_galah', 
        'error': np.nan, 
        'cat': 'galah'
    }
]

fem_cols = [
    {
        'value': 'fem_gspspec', 
        'error': {
            'upper': 'fem_gspspec_upper',
            'lower': 'fem_gspspec_lower'
        }, 
        'cat': 'gspspec'
    }
]

feh_cols = [
    {
        'value': 'feh_galah', 
        'error': np.nan, 
        'cat': 'galah'
    },
    {
        'value': 'feh_lamost', 
        'error': 'e_feh_lamost', 
        'cat': 'lamost'
    },
    {
        'value': 'feh_apogee', 
        'error': 'e_feh_apogee', 
        'cat': 'apogee'
    }
]

# Extract which columns should be used

# par_cols: {
#   value: string
#   error?: {
#     upper: string
#     lower: string
#   } | string
#   cat: string
# }[]
# row: pandas.core.series.Series
def extract_pars(par_cols, row):
    pars = []
    for col in par_cols:
        par = {}
        if np.isnan(row[col['value']]):
            continue
        par['value'] = row[col['value']]
        if type(col['error']) == dict:
            if ('lower' not in col['error']) or ('upper' not in col['error']):
                raise TypeError('lower and/or upper does not exist in error dict')
            par['error'] = {'lower': par['value'] - row[col['error']['lower']] , 'upper': row[col['error']['upper']] - par['value']}
        elif (type(col['error']) == str):
            par['error'] = {'lower': row[col['error']], 'upper': row[col['error']]}
        else:
            par['error'] = {'lower': np.nan, 'upper': np.nan}
        par['cat'] = col['cat']
        pars.append(par)
    return np.array(pars)

# function to select the BEST parameter, not optimized tho, but works

# TypeScript-like style type definitions
# Input
## pars: {
##   value: number
##   error: {
##     lower: number
##     upper: number
##   } | number // upper and lower error OR just singgle number
##   cat: string // catalog name
## }[] // parameter values
## index: number // row index

# Output: Array
## par: {
##   value: number
##   error: {
##     lower: number
##     upper: number
##   } // best upper and lower error
##   symmetric: boolean // if lower == upper
##   cat: string // best catalog name
## }

NaN = {'value': np.nan, 'error': {'lower': np.nan, 'upper': np.nan}, 'symmetric': np.nan , 'cat': np.nan}

def select_best(pars):
    if len(pars) == 0: return NaN
    elif len(pars) == 1:
        if (type(pars[0]['error']) == dict) and ('lower' in pars[0]['error']) and ('upper' in pars[0]['error']):
            if np.isnan(pars[0]['error']['lower']):
                pars[0]['symmetric'] = np.nan
            else:
                pars[0]['symmetric'] = pars[0]['error']['lower'] == pars[0]['error']['upper']
        else:
            raise TypeError('error is not a dict')
        # elif np.isnan(pars[0]['error']):
        #     pars[0]['symmetric'] = np.nan
        # elif (type(pars[0]['error']) == float) or (type(pars[0]['error']) == int):
        #     pars[0]['symmetric'] = True
        # if np.isnan(pars[0]['value']):
        #     pars[0] = NaN
        return pars[0]
    else:
        mask = []
        for i, par in enumerate(pars):
            if (type(par['error']) == dict) and ('lower' in par['error']) or ('upper' in par['error']): 
                mask.append(par['error']['lower'] > 0 and par['error']['upper'] > 0)
            else:
                raise TypeError('error is not a dict')
        pars = pars[mask]
        errors = np.array(list(map(lambda x: (x['error']['lower'] + x['error']['upper'])/2, pars)))
        values = np.array(list(map(lambda x: x['value'], pars)))
        if len(pars) == 0: return NaN
        elif len(pars) == 1:
            pars[0]['symmetric'] = pars[0]['error']['lower'] == pars[0]['error']['upper']
            return pars[0]
        elif len(pars) == 2:
            i = np.argmin(errors)
            pars[i]['symmetric'] = pars[i]['error']['lower'] == pars[i]['error']['upper']
            return pars[i]
        else:
            avg = np.average(values, weights=1/errors)
            selected_pars = []
            for par in pars:
                if (par['value'] + par['error']['upper'] > avg) and (par['value'] - par['error']['lower'] < avg):
                    selected_pars.append(par)
            if (len(selected_pars) == 0):
                i = np.argmin(errors)
                pars[i]['symmetric'] = pars[i]['error']['lower'] == pars[i]['error']['upper']
                return pars[i]
            else:
                errors = np.array(list(map(lambda x: (x['error']['lower'] + x['error']['upper'])/2, selected_pars)))
                i = np.argmin(errors)
                selected_pars[i]['symmetric'] = selected_pars[i]['error']['lower'] == selected_pars[i]['error']['upper']
                return selected_pars[i]
            
remove_cols = ['rv_gaia', 'e_rv_gaia', 'teff_gspphot', 'teff_gspphot_lower', 'teff_gspphot_upper', 
               'logg_gspphot', 'logg_gspphot_lower', 'logg_gspphot_upper', 'mh_gspphot', 
               'mh_gspphot_lower', 'mh_gspphot_upper',  
               'mh_gspspec', 'mh_gspspec_lower', 'mh_gspspec_upper', 'alphafe_gspspec', 'alphafe_gspspec_lower', 
               'alphafe_gspspec_upper', 'fem_gspspec', 'fem_gspspec_lower', 'fem_gspspec_upper',  'rv_rave', 
               'e_rv_rave', 'teff_rave', 'logg_rave', 'mh_rave', 'alphafe_rave', 'rv_galah', 'e_rv_galah', 
               'feh_galah', 'alphafe_galah', 'teff_galah', 'e_teff_galah', 'logg_galah', 'e_logg_galah', 
               'teff_lamost', 'e_teff_lamost', 'logg_lamost', 'e_logg_lamost', 'feh_lamost', 'e_feh_lamost', 
               'rv_lamost', 'e_rv_lamost', 'alpham_lamost', 'e_alpham_lamost', 'rv_apogee', 'e_rv_apogee', 
               'teff_apogee', 'e_teff_apogee', 'logg_apogee', 'e_logg_apogee', 'mh_apogee', 'e_mh_apogee', 
               'alpham_apogee', 'e_alpham_apogee', 'feh_apogee', 'e_feh_apogee']

from time import time
def progressbar(percent=0, width=50, info="", path="", flush=False) -> None:
    left = int((width * percent) // 100)
    right = width - left
    
    tags = "#" * left
    spaces = " " * right
    percents = f"{percent:.0f}%"
    text = f"\r[{tags}{spaces}] {percents} {info}"
    if(flush):
        print(text, end="", flush=True)
    else:
        print(text)
    if(path != ""):
        with open(path, 'a') as f:
            f.write(f"{text}")
            
for file in files[118:]:
    df_vaex = vaex.open(file)
    rvs = []
    teffs = []
    loggs = []
    mhs = []
    alphafes = []
    fems = []
    fehs = []
    t0 = time()
    df = df_vaex.to_pandas_df()
    for index, row in df.iterrows():
        # rv
        rv = extract_pars(rv_cols, row)
        rv_best = select_best(rv)
        rvs.append(rv_best)
        # teff
        teff = extract_pars(teff_cols, row)
        teff_best = select_best(teff)
        teffs.append(teff_best)
        # logg
        logg = extract_pars(logg_cols, row)
        logg_best = select_best(logg)
        loggs.append(logg_best)
        # mh
        mh = extract_pars(mh_cols, row)
        mh_best = select_best(mh)
        mhs.append(mh_best)
        # alphafe
        alphafe = extract_pars(alphafe_cols, row)
        alphafe_best = select_best(alphafe)
        alphafes.append(alphafe_best)
        # fem
        fem = extract_pars(fem_cols, row)
        fem_best = select_best(fem)
        fems.append(fem_best)
        # feh
        feh = extract_pars(feh_cols, row)
        feh_best = select_best(feh)
        fehs.append(feh_best)
        t1 = time()
        progressbar((index+1)/(len(df))*100, flush=True, info=f"{index}/{len(df)} - {np.round(t1-t0, 3)}s - {file.split('/')[-1]}")
    df['rv'] = list(map(lambda x: x['value'], rvs))
    df['e_rv'] = list(map(lambda x: x['error']['lower'], rvs))
    df['rv_cat'] = list(map(lambda x: x['cat'], rvs))
    # teff   
    df['teff'] = list(map(lambda x: x['value'], teffs))
    df['e_teff_lower'] = list(map(lambda x: x['error']['lower'], teffs))
    df['e_teff_upper'] = list(map(lambda x: x['error']['upper'], teffs))
    df['teff_symmetric'] = np.array(list(map(lambda x: x['symmetric'], teffs)))
    df['teff_cat'] = list(map(lambda x: x['cat'], teffs))
    # logg
    df['logg'] = list(map(lambda x: x['value'], loggs))
    df['e_logg_lower'] = list(map(lambda x: x['error']['lower'], loggs))
    df['e_logg_upper'] = list(map(lambda x: x['error']['upper'], loggs))
    df['logg_symmetric'] = np.array(list(map(lambda x: x['symmetric'], loggs)))
    df['logg_cat'] = list(map(lambda x: x['cat'], loggs))
    # mh
    df['mh'] = list(map(lambda x: x['value'], mhs))
    df['e_mh_lower'] = list(map(lambda x: x['error']['lower'], mhs))
    df['e_mh_upper'] = list(map(lambda x: x['error']['upper'], mhs))
    df['mh_symmetric'] = np.array(list(map(lambda x: x['symmetric'], mhs)))
    df['mh_cat'] = list(map(lambda x: x['cat'], mhs))
    # alphafe
    df['alphafe'] = list(map(lambda x: x['value'], alphafes))
    df['e_alphafe_lower'] = list(map(lambda x: x['error']['lower'], alphafes))
    df['e_alphafe_upper'] = list(map(lambda x: x['error']['upper'], alphafes))
    df['alphafe_symmetric'] = np.array(list(map(lambda x: x['symmetric'], alphafes)))
    df['alphafe_cat'] = list(map(lambda x: x['cat'], alphafes))
    # fem
    df['fem'] = list(map(lambda x: x['value'], fems))
    df['e_fem_lower'] = list(map(lambda x: x['error']['lower'], fems))
    df['e_fem_upper'] = list(map(lambda x: x['error']['upper'], fems))
    df['fem_symmetric'] = np.array(list(map(lambda x: x['symmetric'], fems)))
    df['fem_cat'] = list(map(lambda x: x['cat'], fems))
    # feh
    df['feh'] = list(map(lambda x: x['value'], fems))
    df['e_feh_lower'] = list(map(lambda x: x['error']['lower'], fehs))
    df['e_feh_upper'] = list(map(lambda x: x['error']['upper'], fehs))
    df['feh_symmetric'] = np.array(list(map(lambda x: x['symmetric'], fehs)))
    df['feh_cat'] = list(map(lambda x: x['cat'], fehs))
    df_vaex = vaex.from_pandas(df)
    df_vaex = df_vaex.drop(remove_cols)
    path = join(data_dir, file.split('/')[-1])
    df_vaex.export_hdf5(path, progress=True)
    print(f"Saving {path}")
