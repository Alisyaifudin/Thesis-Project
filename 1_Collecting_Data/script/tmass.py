###################################################################
import pathlib
import numpy as np
import vaex
from astroquery.utils.tap.core import Tap

from datetime import datetime
from os.path import join, abspath
from glob import glob
import sys
###################################################################
# import utils
current = pathlib.Path(__file__).parent.resolve()
root_dir = join(current, "..", "..")
if not root_dir in sys.path:
    sys.path.insert(0, root_dir)
from utils import (launch_job, append_name, safe_mkdir,
                   delete_directory, check_df)
###################################################################
# get the root of data directory
root_data_dir = abspath(join(root_dir, "Data"))
# Do the same for 2MASS data
ra_name = "twomass-additional-2"
tmass_data_dir = join(root_data_dir, ra_name)
safe_mkdir(tmass_data_dir)
###################################################################
# 2MASS tap endpoint
tap_tmass = Tap(url="https://irsa.ipac.caltech.edu/TAP/sync")

columns_tmass = ["ra", "dec", "designation", "use_src", "rd_flg"]

# rename the table columns as
columns_tmass_names = ["ra", "dec", "designation", "use_src", "rd_flg"]

tmass_table = "fp_psc"
column_tmass = list(map(lambda x: append_name(x, tmass_table), columns_tmass))
###################################################################


def iterate_job(ras, decs, gen_tmass_query, path_tmass,
                columns_tmass_names, tmass_top=100_000, timeout=1200,
                start_dec=-999, num_tries=10):
    """
    Iterate through the RAs and Decs to query the  TMASS databases
    Args:
        ras (array-like): the RAs to iterate through
        decs (array-like): the Decs to iterate through
        gen_tmass_query (str): the query to run on the TMASS database
        path_tmass (str): the path to save the TMASS dataframes
        columns_tmass_names (list): the columns to query from the TMASS database
        tmass_top (int): the top number of rows to query from the TMASS database
        timeout (int): the timeout for the query
        start_dec (float): the starting dec to start the query from for the first ra
        num_tries (int): the number of tries to query the database before giving up
    """
    old_tmass_top = tmass_top # keep track of the old top for recording purposes
    force_break = False # keep track the force_break flag
    
    # loop through the RAs
    for i, (ra0, ra1) in enumerate(zip(ras[:-1], ras[1:])):
        if force_break:
            break
        # create the directory to save the dataframes
        ra_name = f"ra_{ra0:03d}-{ra1:03d}"
        safe_mkdir(join(path_tmass, ra_name))
        # loop through the Decs
        date0 = datetime.now()
        print(f"RA: {ra0:03d}-{ra1:03d} {date0}")
        for j, (dec0, dec1) in enumerate(zip(decs[:-1], decs[1:])):
            d0 = datetime.now()
            print(f"\tDEC: {dec0:02d}-{dec1:02d} {d0}")
            tries = 0 # keep track of the number of tries
            # if the starting dec is not -999, then we are continuing from the previous query
            if (ra0 == ras[0]) and (dec0 < start_dec):
                continue
            if force_break:
                break
            # loop over the queries until we get a dataframe that is not capped or empty or failed
            while True:
                try_log = f"RA:{ra0:03}-{ra1:03}, DEC:({dec0:02d})-({dec1:02d})"
                tries += 1  # increment the number of tries
                # if we have tried too many times, then something is wrong. Let's force the break
                if tries > num_tries:
                    force_break = True
                # sql query to query the TMASS database
                
                query_tmass = gen_tmass_query(ra0, ra1, dec0, dec1, tmass_top)
                # launch the job to query the TMASS database
                df_tmass = launch_job(
                    tap_tmass.launch_job, query_tmass, cols=columns_tmass_names, duration=timeout)
                res_tmass = check_df(df_tmass, tmass_top)  # check the result
                # if something is wrong, let's force the break
                force_break = res_tmass['force_break']
                if force_break:
                    print(f"\t2MASS {try_log} is empty")
                    break
                # if failed, let's retry with a larger top
                retry = res_tmass['retry']
                old_tmass_top = res_tmass['prev_top']
                tmass_top = res_tmass['new_top']
                if retry:
                    print(
                        f"\t2MASS {try_log} is capped at {old_tmass_top}, increasing to {tmass_top}")
                    continue
                print(
                    f"\t2MASS {try_log} has {len(df_tmass)} rows | TOP={tmass_top}")
                df_tmass.export(
                    join(path_tmass, ra_name, f"dec_({dec0})-({dec1}).hdf5"))
                break
        if force_break:
            break
        # after looping for all decs, let's combine them# do the same for the tmass only
        df_tmass = vaex.open_many(
            glob(join(path_tmass, ra_name, "*.hdf5")))
        df_tmass.export(
            join(path_tmass, f"tmass-{ra0:03d}-{ra1:03d}.hdf5"))
        check_delete = delete_directory(join(path_tmass, ra_name))
        if not check_delete:
            force_break = True
            print(join(path_tmass, f"{ra_name}.txt"),
                  f"Error deleting directory {ra_name}! Stopping loop.")
            # point_in_time(time0)
            break
        print(
            f"tmass RA:{ra0:03}-{ra1:03} is complete with rows {len(df_tmass)}\n")


        # loop again over the next ra
############################################################################################################
if len(sys.argv) < 5 or len(sys.argv) > 6:
    sys.exit("Usage: python tmass.py ra0 ra1 dec0 dec1 (start_dec)")
ra0 = float(sys.argv[1])
ra1 = float(sys.argv[2])
dec0 = float(sys.argv[3])
dec1 = float(sys.argv[4])
start_dec = float(sys.argv[5]) if len(sys.argv) == 6 else -999

ras = np.arange(ra0, ra1+0.1, 1).astype(int)
decs = np.arange(dec0, dec1+0.1, 1).astype(int)


def gen_tmass_query(ra0, ra1, dec0, dec1, top):
    return f"""SELECT TOP {top} {", ".join(columns_tmass)} 
    FROM {tmass_table}
    WHERE ra BETWEEN {ra0} AND {ra1}
    AND dec BETWEEN {dec0} AND {dec1}
    """


print(ras, decs)
############################################################################################################
iterate_job(ras, decs, gen_tmass_query, tmass_data_dir,
            columns_tmass_names, tmass_top=100_0, timeout=900,
            start_dec=start_dec, num_tries=10)
