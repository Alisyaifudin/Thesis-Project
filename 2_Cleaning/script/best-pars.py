import vaex
import numpy as np
from glob import glob
from time import time
from os.path import join, abspath
from os import pardir
from tqdm import tqdm
import pathlib
import pandas as pd
from time import time
from datetime import datetime
import sys

print("initiating...")
current = pathlib.Path(__file__).parent.resolve()
# import utils
root_dir = abspath(join(current, pardir, pardir))
# root data dir
root_data_dir = abspath(join(root_dir, "Data"))
# create directory for the best parameters
name = "Best-Pars"
data_dir = join(root_data_dir, name)
# load combined data
name = "GAIA"
combine_data_dir = join(root_data_dir, name)
# see the data
files = glob(join(combine_data_dir, "*.hdf5"))
files.sort()
############################################
# utility functions
############################################
print("function...")

sym_ass = {
    True: 1,
    False: 0,
    None: -1
}


def assign(cat, keys):
    return [cat[key] for key in keys]


def best_pars(row, catalogs, par="p"):
    best_val = np.nan
    best_cat = np.nan
    best_el = np.nan
    best_eu = np.nan
    best_sym = np.nan
    keys = ["name", "value", "e_value_lower", "e_value_upper", "sym"]
    cats = []
    for catalog in catalogs:
        c = {
            "name": catalog['name'],
            "value": row[catalog['value']] if catalog['value'] != None else np.nan,
            "e_value_upper": row[catalog['e_value_upper']] if catalog['e_value_upper'] != None else np.nan,
            "e_value_lower": row[catalog['e_value_lower']] if catalog['e_value_lower'] != None else np.nan,
            "e_effective": row[catalog['e_value_upper']] if catalog['e_value_upper'] != None else np.nan,
            "sym": sym_ass[catalog['sym']]
        }
        if catalog['sym'] == False:
            c['e_value_upper'] = c['e_value_upper'] - c['value']
            c['e_value_lower'] = c['value'] - c['e_value_lower']
            c['e_effective'] = np.sqrt(
                [c['e_value_upper']*c['e_value_lower']][0])
        if c['e_effective'] == 0:
            c['e_effective'] = 1000000
        cats.append(c)
    cats_filtered = [c for c in cats if not np.isnan(c['value'])]

    if len(cats_filtered) == 1:
        cat = cats_filtered[0]
        best_cat, best_val, best_el, best_eu, best_sym = assign(cat, keys)
    elif len(cats_filtered) > 1:
        cats_filtered_2 = [
            c for c in cats_filtered if not np.isnan(c['e_value_upper'])]
        if len(cats_filtered_2) == 1:
            cat = cats_filtered_2[0]
            best_cat, best_val, best_el, best_eu, best_sym = assign(cat, keys)
        elif len(cats_filtered_2) == 2:
            minarg = np.argmin([c['e_effective'] for c in cats_filtered_2])
            cat = cats_filtered_2[minarg]
            best_cat, best_val, best_el, best_eu, best_sym = assign(cat, keys)
        elif len(cats_filtered_2) > 2:
            weights = np.array(
                [1/c['e_effective']**2 for c in cats_filtered_2])
            avg = np.average([c['value']
                             for c in cats_filtered_2], weights=weights)
            m = 2
            mask = (avg > np.array([c['value']-m*c['e_value_lower'] for c in cats_filtered_2])) * (
                avg < np.array([c['value']+m*c['e_value_upper'] for c in cats_filtered_2]))
            cats_filtered_3 = np.array(cats_filtered_2)[mask]
            if len(cats_filtered_3) == 0:
                minarg = np.argmin([c['e_effective'] for c in cats_filtered_2])
                cat = cats_filtered_2[minarg]
                best_cat, best_val, best_el, best_eu, best_sym = assign(
                    cat, keys)
            elif len(cats_filtered_3) == 1:
                cat = cats_filtered_3[0]
                best_cat, best_val, best_el, best_eu, best_sym = assign(
                    cat, keys)
            else:
                minarg = np.argmin([c['e_effective'] for c in cats_filtered_3])
                cat = cats_filtered_3[minarg]
                best_cat, best_val, best_el, best_eu, best_sym = assign(
                    cat, keys)
    row[f'{par}'] = best_val
    row[f'{par}_cat'] = best_cat
    row[f'{par}_el'] = best_el
    row[f'{par}_eu'] = best_eu
    row[f'{par}_sym'] = best_sym
    return row


############################################
print("catalogs...")
catalogs_rv = [
    {"name": "gaia", "value": "rv_gaia", "e_value_upper": "e_rv_gaia",
        "e_value_lower": "e_rv_gaia", "sym": True},
    {"name": "rave", "value": "rv_rave", "e_value_upper": "e_rv_rave",
        "e_value_lower": "e_rv_rave", "sym": True},
    {"name": "galah", "value": "rv_galah", "e_value_upper": "e_rv_galah",
        "e_value_lower": "e_rv_galah", "sym": True},
    {"name": "lamost", "value": "rv_lamost", "e_value_upper": "e_rv_lamost",
        "e_value_lower": "e_rv_lamost", "sym": True},
    {"name": "apogee", "value": "rv_apogee", "e_value_upper": "e_rv_apogee",
        "e_value_lower": "e_rv_apogee", "sym": True},
]

catalogs_teff = [
    {"name": "gspphot", "value": "teff_gspphot", "e_value_upper": "teff_gspphot_upper",
        "e_value_lower": "teff_gspphot_lower", "sym": False},
    {"name": "rave", "value": "teff_rave", "e_value_upper": None,
        "e_value_lower": None, "sym": None},
    {"name": "galah", "value": "teff_galah", "e_value_upper": "e_teff_galah",
        "e_value_lower": "e_teff_galah", "sym": True},
    {"name": "lamost", "value": "teff_lamost", "e_value_upper": "e_teff_lamost",
        "e_value_lower": "e_teff_lamost", "sym": True},
    {"name": "apogee", "value": "teff_apogee", "e_value_upper": "e_teff_apogee",
        "e_value_lower": "e_teff_apogee", "sym": True},
]
catalogs_logg = [
    {"name": "gspphot", "value": "logg_gspphot", "e_value_upper": "logg_gspphot_upper",
        "e_value_lower": "logg_gspphot_lower", "sym": False},
    {"name": "rave", "value": "logg_rave", "e_value_upper": None,
        "e_value_lower": None, "sym": None},
    {"name": "galah", "value": "logg_galah", "e_value_upper": "e_logg_galah",
        "e_value_lower": "e_logg_galah", "sym": True},
    {"name": "lamost", "value": "logg_lamost", "e_value_upper": "e_logg_lamost",
        "e_value_lower": "e_logg_lamost", "sym": True},
    {"name": "apogee", "value": "logg_apogee", "e_value_upper": "e_logg_apogee",
        "e_value_lower": "e_logg_apogee", "sym": True},
]
catalogs_feh = [
    {"name": "galah", "value": "feh_galah", "e_value_upper": "e_feh_galah",
        "e_value_lower": "e_feh_galah", "sym": True},
    {"name": "lamost", "value": "feh_lamost", "e_value_upper": "e_feh_lamost",
        "e_value_lower": "e_feh_lamost", "sym": True},
    {"name": "apogee", "value": "feh_apogee", "e_value_upper": "e_feh_apogee",
        "e_value_lower": "e_feh_apogee", "sym": True},
]

catalogs_mh = [
    {"name": "gspphot", "value": "mh_gspphot", "e_value_upper": "mh_gspphot_upper",
        "e_value_lower": "mh_gspphot_lower", "sym": False},
    {"name": "gspspec", "value": "mh_gspspec", "e_value_upper": "mh_gspspec_upper",
        "e_value_lower": "mh_gspspec_lower", "sym": False},
    {"name": "rave", "value": "mh_rave", "e_value_upper": None,
        "e_value_lower": None, "sym": None},
    {"name": "apogee", "value": "mh_apogee", "e_value_upper": "e_mh_apogee",
        "e_value_lower": "e_mh_apogee", "sym": True},
]

catalogs_alphafe = [
    {"name": "gspspec", "value": "alphafe_gspspec", "e_value_upper": "alphafe_gspspec_upper",
        "e_value_lower": "alphafe_gspspec_lower", "sym": False},
    {"name": "rave", "value": "alphafe_rave",
        "e_value_upper": None, "e_value_lower": None, "sym": None},
    {"name": "galah", "value": "alphafe_galah", "e_value_upper": "e_alphafe_galah",
        "e_value_lower": "e_alphafe_galah", "sym": True},
]

catalogs_alpham = [
    {"name": "lamost", "value": "alpham_lamost", "e_value_upper": "e_alpham_lamost",
        "e_value_lower": "e_alpham_lamost", "sym": True},
    {"name": "apogee", "value": "alpham_apogee", "e_value_upper": "e_alpham_apogee",
        "e_value_lower": "e_alpham_apogee", "sym": True},
]

catalogs = {
    "rv": catalogs_rv,
    "teff": catalogs_teff,
    "logg": catalogs_logg,
    "feh": catalogs_feh,
    "mh": catalogs_mh,
    "alphafe": catalogs_alphafe,
    "alpham": catalogs_alpham,
}
print("loop...")

if len(sys.argv) != 3:
    sys.exit("Usage: python3 loop.py <from> <to>")
fr = int(sys.argv[1])
to = int(sys.argv[2])

# python -u best-pars.py  | tee log/log-

for file in files[fr:to]:
    t0 = time()
    name = file.split("/")[-1]
    print(name)
    df_vaex = vaex.open(file)
    df_vaex = df_vaex
    t0 = time()
    df = df_vaex.to_pandas_df()
    # df = df[:4000]
    # create empty dataframe
    df_com = pd.DataFrame()
    M = 100
    # iterate dataframe into chunks
    tot = len(df)//M
    cols = np.array([])
    for catalog in catalogs.values():
        col = np.array([[c['value'], c["e_value_upper"], c["e_value_lower"]]
                       for c in catalog]).flatten()
        col = np.unique(col[col != None])
        cols = np.append(cols, col)
    for i, df_chunk in tqdm(df.groupby(df.index // M)):
        # print(i, len(df_chunk))
        # iterate chunks into catalogs
        for par, catalog in catalogs.items():
            df_chunk = df_chunk.apply(best_pars, axis=1, args=(catalog, par))
        if len(df_com) == 0:
            df_com = df_chunk
        else:
            df_com = pd.concat([df_com, df_chunk])
        # progressbar(i/tot*100, info=f"{np.round(t1-t0,2)} s", flush=True)
    df_com.drop(columns=cols, inplace=True)
    # df = df.apply(best_pars, axis=1, args=(catalogs_rv, "rv"))
    df = vaex.from_pandas(df_com)
    df.export(join(data_dir, name), progress=True)
    t1 = time()
    print(f"Time: {np.round(t1-t0,2)} s")
    print(f"Saved {name} | {len(df_com)} rows | {datetime.now()}")
