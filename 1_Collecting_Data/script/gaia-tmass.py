###################################################################
import pathlib
import numpy as np
import vaex
from astroquery.gaia import Gaia
from astroquery.utils.tap.core import Tap

from datetime import datetime
from time import time
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
# Create a directory for Gaia DR3 and 2MASS data or if it already exists, just move on
ra_name = "Gaia-all"
gaia_data_dir = join(root_data_dir, ra_name)
safe_mkdir(gaia_data_dir)
# Do the same for 2MASS data
ra_name = "twomass-all"
tmass_data_dir = join(root_data_dir, ra_name)
safe_mkdir(tmass_data_dir)
###################################################################
column_gaia = [
    "source_id",
    "pm", "pmra", "pmra_error AS e_pmra", "pmdec", "pmdec_error AS e_pmdec",
    "parallax", "parallax_error AS e_parallax",
    "phot_g_mean_mag AS Gmag",	"phot_bp_mean_mag AS BPmag", "phot_rp_mean_mag AS RPmag",
    "phot_bp_mean_flux_over_error AS Fb_over_err",
    "phot_rp_mean_flux_over_error AS Fr_over_err",
    "ruwe",
    "phot_bp_rp_excess_factor AS excess_factor",
    "radial_velocity AS rv_gaia", "radial_velocity_error AS e_rv_gaia",
    "l AS glon", "b AS glat",
    "teff_gspphot", "teff_gspphot_lower", "teff_gspphot_upper",
    "logg_gspphot", "logg_gspphot_lower", "logg_gspphot_upper"
]

column_astrophysical = [
    "mh_gspphot", "mh_gspphot_lower", "mh_gspphot_upper",
    "distance_gspphot", "distance_gspphot_lower", "distance_gspphot_upper",
    "ag_gspphot", "ag_gspphot_lower", "ag_gspphot_upper",
    "mh_gspspec", "mh_gspspec_lower", "mh_gspspec_upper",
    "alphafe_gspspec", "alphafe_gspspec_lower", "alphafe_gspspec_upper",
    "fem_gspspec", "fem_gspspec_lower", "fem_gspspec_upper",
    "spectraltype_esphs"
]

column_join = ["original_psc_source_id AS tmass"]

gaia_alias = "gdr3"
astrophysical_alias = "astrophysical"
join_alias = "join_table"

column_gaia = list(map(lambda x: append_name(x, gaia_alias), column_gaia))
column_astrophysical = list(map(lambda x: append_name(
    x, astrophysical_alias), column_astrophysical))
column_join = list(map(lambda x: append_name(x, join_alias), column_join))

columns = column_gaia + column_astrophysical + column_join
###################################################################
# 2MASS tap endpoint
tap_tmass = Tap(url="https://irsa.ipac.caltech.edu/TAP/sync")

columns_tmass = ["ra", "dec","j_m", "k_m", "designation", "ph_qual", "use_src", "rd_flg"]

# rename the table columns as 
columns_tmass_names = ["ra", "dec","Jmag", "Kmag", "designation", "ph_qual", "use_src", "rd_flg"]

tmass_table = "fp_psc"
column_tmass = list(map(lambda x: append_name(x, tmass_table), columns_tmass))
###################################################################


def iterate_job(ras, decs, gen_gaia_query, gen_tmass_query, path_gaia, path_tmass,
                columns_tmass_names, gaia_top=100_000, tmass_top=100_000, timeout=1200,
                start_dec=-999, num_tries=10):
    """
    Iterate through the RAs and Decs to query the Gaia and TMASS databases
    Args:
        ras (array-like): the RAs to iterate through
        decs (array-like): the Decs to iterate through
        gen_gaia_query (str): the query to run on the Gaia database
        gen_tmass_query (str): the query to run on the TMASS database
        path_gaia (str): the path to save the Gaia dataframes
        path_tmass (str): the path to save the TMASS dataframes
        columns_tmass_names (list): the columns to query from the TMASS database
        gaia_top (int): the top number of rows to query from the Gaia database
        tmass_top (int): the top number of rows to query from the TMASS database
        timeout (int): the timeout for the query
        start_dec (float): the starting dec to start the query from for the first ra
        num_tries (int): the number of tries to query the database before giving up
    """
    # keep track of the old top for recording purposes
    old_gaia_top = gaia_top
    old_tmass_top = tmass_top
    # keep track the force_break flag
    force_break = False
    # loop through the RAs
    for i, (ra0, ra1) in enumerate(zip(ras[:-1], ras[1:])):
        if force_break:
            break
        # create the directory to save the dataframes
        ra_name = f"ra_{ra0:03d}-{ra1:03d}"
        safe_mkdir(join(path_gaia, ra_name))
        safe_mkdir(join(path_tmass, ra_name))
        # loop through the Decs
        time0 = time()
        date0 = datetime.now()
        print(f"RA: {ra0:03d}-{ra1:03d} {date0}")
        for j, (dec0, dec1) in enumerate(zip(decs[:-1], decs[1:])):
            t0 = time()
            d0 = datetime.now()
            print(f"\tDEC: {dec0:02d}-{dec1:02d} {d0}")
            # keep track of the number of tries
            tries = 0
            # if the starting dec is not -999, then we are continuing from the previous query
            if (ra0 == ras[0]) and (dec0 < start_dec):
                continue
            if force_break:
                break
            # loop over the queries until we get a dataframe that is not capped or empty or failed
            while True:
                try_log = f"RA:{ra0:03}-{ra1:03}, DEC:({dec0:02d})-({dec1:02d})"
                # increment the number of tries
                tries += 1
                # if we have tried too many times, then something is wrong. Let's force the break
                if tries > num_tries:
                    force_break = True
                    print(
                        f"\n\tGaia {try_log} have tried {num_tries} times, something wrong")
                    # point_in_time(t0)
                if force_break:
                    break
                # sql query to query the Gaia database
                query_gaia = gen_gaia_query(ra0, ra1, dec0, dec1, gaia_top)
                # launch the job to query the Gaia database
                df_gaia = launch_job(
                    Gaia.launch_job, query_gaia, duration=timeout)
                # check the result
                res_gaia = check_df(df_gaia, gaia_top)
                # if something is wrong, let's force the break
                force_break = res_gaia['force_break']
                if force_break:
                    print(f"\n\tGaia {try_log} is empty")
                    # point_in_time(t0)
                    break
                # if failed, let's retry with a larger top
                retry = res_gaia['retry']
                old_gaia_top = res_gaia['prev_top']
                gaia_top = res_gaia['new_top']
                if retry:
                    print(
                        f"\n\tGaia  {try_log} is capped at {old_gaia_top}, increasing to {gaia_top}")
                    # point_in_time(t0)
                    continue
                # if everything is fine, let's continue the journey
                print(
                    f"\n\tGaia {try_log} has {len(df_gaia)} rows | TOP={gaia_top}")
                # point_in_time(t0)
                # sql query to query the TMASS database
                query_tmass = gen_tmass_query(ra0, ra1, dec0, dec1, tmass_top)
                # launch the job to query the TMASS database
                df_tmass = launch_job(
                    tap_tmass.launch_job, query_tmass, cols=columns_tmass_names, duration=timeout)
                # check the result
                res_tmass = check_df(df_tmass, tmass_top)
                # if something is wrong, let's force the break
                force_break = res_tmass['force_break']
                if force_break:
                    print(f"\t2MASS {try_log} is empty")
                    # point_in_time(t0)
                    break
                # if failed, let's retry with a larger top
                retry = res_tmass['retry']
                old_tmass_top = res_tmass['prev_top']
                tmass_top = res_tmass['new_top']
                if retry:
                    print(
                        f"\t2MASS {try_log} is capped at {old_tmass_top}, increasing to {tmass_top}")
                    # point_in_time(t0)
                    continue
                # if everything is fine, let's continue the journey
                print(
                    f"\t2MASS {try_log} has {len(df_tmass)} rows | TOP={tmass_top}")
                # point_in_time(t0)
                # ===========================
                # this is to get the tmass only
                # because we are querying a larger area, we need to filter the result
                df_tmass_only = df_tmass.filter(f"ra > {ra0}").filter(
                    f"ra < {ra1}").filter(f"dec > {dec0}").filter(f"dec < {dec1}")
                df_tmass_only = df_tmass_only.extract()
                # to avoid this random stup*d error
                # RuntimeError: Oops, get an empty chunk, from 1024 to 1024, that should not happen
                # convert to pandas, then back to vaex
                a = df_tmass_only.to_pandas_df()
                df_tmass_only = vaex.from_pandas(a)
                # ===========================
                # join the gaia and tmass
                df_join = df_gaia.join(
                    df_tmass, right_on="designation", left_on="tmass", how="left")
                # delete the columns that are not needed
                df_join.drop(columns=["designation", "tmass"], inplace=True)
                # filtered nan in joining
                df_join = df_join.filter("ra > 0.")
                df_join = df_join.extract()
                # save the result to the disk
                df_join.export(
                    join(path_gaia, ra_name, f"dec_({dec0})-({dec1}).hdf5"))
                print(
                    f"\tgaia&tmass RA:{ra0:03}-{ra1:03}, DEC:({dec0})-({dec1}) is complete with rows {len(df_join)}")
                # point_in_time(t0)
                df_tmass_only.export(
                    join(path_tmass, ra_name, f"dec_({dec0})-({dec1}).hdf5"))
                # do the same for the tmass only
                print(
                    f"\ttmass ONLY RA:{ra0:03}-{ra1:03}, DEC:({dec0})-({dec1}) is complete with rows {len(df_tmass_only)}")
                # point_in_time(t0)
                # get out of the while loop
                break
        # after looping for all decs, let's combine them
        df_gaia_tmass = vaex.open_many(
            glob(join(path_gaia, ra_name, "*.hdf5")))
        # save
        df_gaia_tmass.export(
            join(path_gaia, f"gaia-{ra0:03d}-{ra1:03d}.hdf5"))
        # delete the temporary directory
        check_delete = delete_directory(join(path_gaia, ra_name))
        # check if the directory is deleted
        if not check_delete:
            force_break = True
            print(join(path_gaia, f"{ra_name}.txt"),
                  f"Error deleting directory {ra_name}! Stopping loop.")
            # point_in_time(time0)
            break
        # tell the world that everything is fine
        print(
            f"gaia&tmass RA:{ra0:03}-{ra1:03} is complete with rows {len(df_gaia_tmass)}")
        # point_in_time(time0)
        # do the same for the tmass only
        df_tmass_only = vaex.open_many(
            glob(join(path_tmass, ra_name, "*.hdf5")))
        df_tmass_only.export(
            join(path_tmass, f"tmass-{ra0:03d}-{ra1:03d}.hdf5"))
        check_delete = delete_directory(join(path_tmass, ra_name))
        if not check_delete:
            force_break = True
            print(join(path_tmass, f"{ra_name}.txt"),
                  f"Error deleting directory {ra_name}! Stopping loop.")
            # point_in_time(time0)
            break
        print(
            f"tmass only RA:{ra0:03}-{ra1:03} is complete with rows {len(df_tmass_only)}\n")

        # point_in_time(time0)
        # loop again over the next ra
############################################################################################################
if len(sys.argv) < 5 or len(sys.argv) > 6:
    sys.exit("Usage: python gaia_tmass.py ra0 ra1 dec0 dec1 (start_dec)")
ra0 = float(sys.argv[1])
ra1 = float(sys.argv[2])
dec0 = float(sys.argv[3])
dec1 = float(sys.argv[4])
start_dec = float(sys.argv[5]) if len(sys.argv) == 6 else -999

ras = np.arange(ra0, ra1+0.1, 1).astype(int)
decs = np.arange(dec0, dec1+0.1, 1).astype(int)


def gen_gaia_query(ra0, ra1, dec0, dec1, top):
    return f"""SELECT TOP {top} {', '.join(columns)}
    FROM gaiadr3.gaia_source AS {gaia_alias}
    LEFT JOIN gaiadr3.astrophysical_parameters AS {astrophysical_alias} ON astrophysical.source_id = gdr3.source_id
    RIGHT JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS tmass ON tmass.source_id = gdr3.source_id
    RIGHT JOIN gaiadr3.tmass_psc_xsc_join as {join_alias} ON join_table.clean_tmass_psc_xsc_oid = tmass.clean_tmass_psc_xsc_oid
    WHERE ra BETWEEN {ra0} AND {ra1} 
    AND dec BETWEEN {dec0} AND {dec1}
    AND parallax is not null
    """


def gen_tmass_query(ra0, ra1, dec0, dec1, top):
    return f"""SELECT TOP {top} {", ".join(columns_tmass)} 
    FROM {tmass_table}
    WHERE ra BETWEEN {ra0-0.5} AND {ra1+0.5}
    AND dec BETWEEN {dec0-0.5} AND {dec1+0.5}
    AND (ph_qual like 'A__' OR (rd_flg like '1__' OR rd_flg like '3__'))
    AND (ph_qual like '__A' OR (rd_flg like '__1' OR rd_flg like '__3')) 
    AND use_src='1' AND ext_key is null
    AND (j_m-k_m) > -0.05
    AND (j_m-k_m) < 1.0
    AND j_m < 13.5 AND j_m > 2
    """


print(ras, decs)
############################################################################################################
iterate_job(ras, decs, gen_gaia_query, gen_tmass_query, gaia_data_dir, tmass_data_dir,
            columns_tmass_names, gaia_top=100_000, tmass_top=100_000, timeout=600,
            start_dec=start_dec, num_tries=10)
