from astroquery.utils.tap.core import Tap
import vaex
import numpy as np
from datetime import datetime
from time import time
from astroquery.gaia import Gaia
from os.path import join, abspath
from os import pardir, mkdir, curdir
from glob import glob
import sys

# import utils
thesis_dir = abspath("../../")
current_dir = abspath(curdir)
sys.path.insert(0, thesis_dir)

from utils import timeout, progressbar, appendName

root_data_dir = abspath(join(thesis_dir, "Data"))

name = "Gaia-2MASS"
gaia_data_dir = join(root_data_dir, name)
# print(gaia_data_dir)
try:
    mkdir(gaia_data_dir)
    print(f"Creating {gaia_data_dir} dir in Data dir")
except FileExistsError:
    print(f"Directory {gaia_data_dir} already exist. Good to go!")

name = "TWOMASS"
tmass_data_dir = join(root_data_dir, name)
try:
    mkdir(tmass_data_dir)
    print(f"Creating {tmass_data_dir} dir in Data dir")
except FileExistsError:
    print(f"Directory {tmass_data_dir} already exist. Good to go!")

column_gaia = ["source_id", "pm", "pmra", "pmra_error AS e_pmra", "pmdec", 
           "pmdec_error AS e_pmdec", "parallax", "parallax_error AS e_parallax", "phot_g_mean_mag AS Gmag",	"phot_bp_mean_mag AS BPmag", 
           "phot_rp_mean_mag AS RPmag", "phot_bp_mean_flux_over_error AS Fb_over_err", "phot_rp_mean_flux_over_error AS Fr_over_err", 
           "ruwe", "phot_bp_rp_excess_factor AS excess_factor", "radial_velocity AS rv_gaia",	"radial_velocity_error AS e_rv_gaia",
           "l AS GLON", "b AS GLAT", "teff_gspphot", "teff_gspphot_lower", "teff_gspphot_upper",
           "logg_gspphot", "logg_gspphot_lower", "logg_gspphot_upper"]

column_astrophysical = ["mh_gspphot", "mh_gspphot_lower", "mh_gspphot_upper", "distance_gspphot", "distance_gspphot_lower", 
                         "distance_gspphot_upper", "ag_gspphot", "ag_gspphot_lower", "ag_gspphot_upper",
                         "mh_gspspec", "mh_gspspec_lower", "mh_gspspec_upper", "alphafe_gspspec", "alphafe_gspspec_lower", 
                         "alphafe_gspspec_upper", "fem_gspspec", "fem_gspspec_lower", "fem_gspspec_upper" ,"spectraltype_esphs"]

column_join = ["original_psc_source_id AS tmass"]

column_gaia = list(map(lambda x: appendName(x, "gdr3"), column_gaia))
column_astrophysical = list(map(lambda x: appendName(x, "astrophysical"), column_astrophysical))
column_join = list(map(lambda x: appendName(x, "join_table"), column_join))

columns = column_gaia + column_astrophysical + column_join

# 2MASS tap endpoint
tap_tmass = Tap(url="https://irsa.ipac.caltech.edu/TAP/sync")

columns_tmass = ["ra", "dec","j_m", "k_m", "designation", "ph_qual"]
columns_tmass_names = ["ra", "dec", "Jmag", "Kmag", "designation", "ph_qual"]

ra_lower = int(sys.argv[1])
ra_upper = int(sys.argv[2])
dec_lower = int(sys.argv[3])
dec_upper = int(sys.argv[4])


ras = np.arange(ra_lower, ra_upper+0.1, 1).astype(int)
dra = ras[1] - ras[0]

print("ra: ", ras)

decs = np.arange(dec_lower, dec_upper+0.1, 1).astype(int)
ddec = decs[1] - decs[0]
print("dec: ", decs)

log_name = sys.argv[5]

ORI_TOP = 50_000_000
# ra0 for lower boundry and ra1 for upper boundary
# same with dec0 and dec1
text_path = join(current_dir, log_name)
for i, (ra0, ra1) in enumerate(zip(ras[:-1], ras[1:])):
    TOP = ORI_TOP # cap maximum rows for each response, so that the response is not exploding
    df_com = [] #initial table
    df_com_tmass = [] #initial tmass table
    time0 = time()
    progressbar(0, info=f"{ra0}-{ra1}")
    j = 0
    skip = False
    trying = 0
    while j < len(decs) -1:
        if trying > 25:
            text = "too many tries, raise error"
            print(text)
            with open(text_path, 'a') as f:
                f.write(f"{text}\n")
            raise Exception("too many tries")
        if ~skip:
            t0 = time()
        dec0 = decs[j]
        dec1 = decs[j+1]
        # query gaia data
        # taking wider ra and dec constrains than 2MASS, because of different epoch
        # the contrains are based on https://doi.org/10.1093/mnras/stab3671
        query_gaia = f"""
        SELECT TOP {TOP} {', '.join(columns)}
        FROM gaiadr3.gaia_source AS gdr3
        LEFT JOIN gaiadr3.astrophysical_parameters AS astrophysical ON astrophysical.source_id = gdr3.source_id
        RIGHT JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS tmass ON tmass.source_id = gdr3.source_id
        RIGHT JOIN gaiadr3.tmass_psc_xsc_join as join_table ON join_table.clean_tmass_psc_xsc_oid = tmass.clean_tmass_psc_xsc_oid
        WHERE gdr3.ra BETWEEN {ra0-dra*0.5} AND {ra1+dra*0.5}
        AND gdr3.dec BETWEEN {dec0-ddec*0.5} AND {dec1+ddec*0.5} 
        """
        job_gaia = timeout(Gaia.launch_job, args=(query_gaia,), timeout_duration=1200, path=text_path)
        if job_gaia == None: #if failed, try again
            text = f"fail to fetch gaia"
            with open(text_path, 'a') as f: 
                f.write(f"{text}\n")
            skip = True
            trying += 1
            continue
        result_gaia = job_gaia.get_results()
        if(len(result_gaia) == TOP):
            text= f"tmass data is capped, increase TOP | old: {TOP} | new: {TOP*2}"
            print(text)
            with open(text_path, 'a') as f:
                f.write(f"{text}\n")
            TOP *= 2
            skip = True
            continue
        df_pandas = result_gaia.to_pandas()
        df_pandas = df_pandas.drop_duplicates(subset=['tmass'], keep="first")
        df_pandas.rename(columns={'glon': 'GLON', 'glat': 'GLAT'}, inplace=True)
        df_gaia = vaex.from_pandas(df_pandas)
        # query 2MASS data
        query_tmass = f"""
        SELECT TOP {TOP} {", ".join(columns_tmass)} 
        FROM fp_psc
        WHERE ra BETWEEN {ra0} AND {ra1}
        AND dec BETWEEN {dec0} AND {dec1} 
        """
        job_tmass = timeout(tap_tmass.launch_job, args=(query_tmass,), timeout_duration=600, path=text_path)
        if job_tmass == None: 
            text = f"fail to fetch tmass"
            with open(text_path, 'a') as f:
                f.write(f"{text}\n")
            skip = True
            trying += 1
            continue
        result_tmass = job_tmass.get_results()
        if(len(result_tmass) == TOP):
            text = f"tmass data is capped, increase TOP | old: {TOP} | new: {TOP*2}"
            print(text)
            with open(text_path, 'a') as f:
                f.write(f"{text}\n")
            TOP *= 2
            skip = True
            continue
        df_tmass = result_tmass.to_pandas()
        df_tmass.columns = columns_tmass_names
        # join
        df_tmass = vaex.from_pandas(df_tmass)
        join_table = df_tmass.join(df_gaia, left_on="designation", right_on="tmass", how="left")
        join_table.drop(["designation", "tmass"], inplace=True)
        if(len(df_com) == 0):
            df_com = join_table
            df_com_tmass = df_tmass
        else:
            df_com = df_com.concat(join_table)
            df_com_tmass = df_com_tmass.concat(df_tmass)
        j += 1
        t1 = time()
        skip = False
        trying = 0
        
        progressbar((j)/(len(decs)-1)*100, path=text_path,info=f"{ra0}-{ra1} | [{dec0}]-[{dec1}] | {round(t1-t0,2)} s | join = {len(join_table)} | tmass = {len(df_tmass)} | gaia = {len(df_gaia)}| TOP = {TOP}")
        TOP = np.max([int(len(df_tmass) * 2), ORI_TOP, int(len(df_gaia) * 2)])
    time1 = time()  
    df_com.export(join(gaia_data_dir, f"gaia-{ra0:03d}-{ra1:03d}.hdf5"), progress=True)
    df_com_tmass.export(join(tmass_data_dir, f"tmass-{ra0:03d}-{ra1:03d}.hdf5"), progress=True)
    print(f"{len(df_com)} || {round((time1-time0)/60, 2)}m")
    print(f"{i} saved {ra0}-{ra1} || {datetime.now()}")
    
