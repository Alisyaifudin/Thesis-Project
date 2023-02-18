from astroquery.utils.tap.core import Tap
import vaex
import numpy as np
from datetime import datetime
from time import time
from astroquery.gaia import Gaia
from os.path import join, abspath
from os import pardir, mkdir
from glob import glob
import sys

# import utils
util_dir = abspath("../../")
sys.path.insert(0, util_dir)

from utils import timeout, progressbar, appendName

root_data_dir = abspath(join(util_dir, "Data"))

name = "GUMS"
gaia_data_dir = join(root_data_dir, name)
try:
    mkdir(gaia_data_dir)
    print(f"Creating {gaia_data_dir} dir in Data dir")
except FileExistsError:
    print(f"Directory {gaia_data_dir} already exist. Good to go!")

columns = ["ra", "dec", "barycentric_distance", "pmra", "pmdec", "radial_velocity",
              "mag_g", "mag_bp", "mag_rp", "feh", "alphafe", "mass", "population", "logg", "teff", "spectral_type"]


# divide into 360 RAs, depend on preference
ras = np.arange(0,360+0.1, 10).astype(int)
dra = ras[1] - ras[0]

decs = np.arange(-90,90+0.1,30).astype(int)
ddec = decs[1] - decs[0]

# ra0 for lower boundry and ra1 for upper boundary
# same with dec0 and dec1
ORI_TOP = 50_000_000
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
        if trying > 15:
            print("too many tries, raise error")
            raise Exception("too many tries")
        if ~skip:
            t0 = time()
        dec0 = decs[j]
        dec1 = decs[j+1]
        # query gaia data
        # taking wider ra and dec constrains than 2MASS, because of different epoch
        query_gaia = f"""
        SELECT TOP {TOP} {', '.join(columns)}
        FROM gaiadr3.gaia_universe_model AS gdr3
        WHERE gdr3.ra BETWEEN {ra0-dra*1} AND {ra1+dra*1}
        AND gdr3.dec BETWEEN {dec0-ddec*1} AND {dec1+ddec*1} 
        AND barycentric_distance < 1000
        """
        job_gaia = timeout(Gaia.launch_job, args=(query_gaia,), timeout_duration=600)
        if job_gaia == None: #if failed, try again
            print("fail to fetch gaia")
            print("length = ", len(df_com))
            skip = True
            trying += 1
            continue
        result_gaia = job_gaia.get_results()
        if(len(result_gaia) == TOP):
            print(f"gaia data is capped, increase TOP | {TOP}")
            TOP *= 2
            skip = True
            continue
        df_pandas = result_gaia.to_pandas()
        df_gaia = vaex.from_pandas(df_pandas)
        if(len(df_com) == 0):
            df_com = df_gaia
        else:
            df_com = df_com.concat(df_gaia)
        j += 1
        t1 = time()
        skip = False
        trying = 0
        TOP = np.max([ORI_TOP, int(len(df_gaia) * 2)])
        progressbar((j)/(len(decs)-1)*100, info=f"{ra0}-{ra1} | [{dec0}]-[{dec1}] | {round(t1-t0,2)} s | gaia = {len(df_gaia)}| TOP = {TOP}")
    time1 = time()  
    df_com.export(join(gaia_data_dir, f"gaia-{ra0:03d}-{ra1:03d}.hdf5"), progress=True)
    print(f"{len(df_com)} || {round((time1-time0)/60, 2)}m")
    print(f"{i} saved {ra0}-{ra1} || {datetime.now()}")
