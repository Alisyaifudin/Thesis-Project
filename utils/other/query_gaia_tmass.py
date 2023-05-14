from utils import print_or_write, safe_mkdir, launch_job, write_to, delete_directory
from os.path import join
import vaex
from astroquery.gaia import Gaia
from astroquery.utils.tap.core import Tap
from glob import glob
from time import time
from datetime import datetime

tap_tmass = Tap(url="https://irsa.ipac.caltech.edu/TAP/sync")

def point_in_time(t0, log_file, write=False):
    """
    Print the time elapsed since the start of the query
    """
    t1 = time()
    d1 = datetime.now()
    print_or_write(f"======= {d1} | {t1-t0} s =======", log_file, write=write)

def check_df(df, top):
    """
    Check if the dataframe is None, capped, or empty

    Args:
        df (DataFrame-like): the dataframe to check
        top (int): the top number of rows to query
    
    Returns:
        (bool, bool, int): (force_break, retry, top, new_top)
    """
    new_top = top # default new top
    if df is None:
        # if the dataframe is None, therefore the query failed for some reason
        return False, True, top, top
    elif len(df) == top:
        # if the dataframe is capped, double the top number of rows to query
        new_top = 2*top
        return False, True, top, new_top
    elif len(df) == 0:
        # if the dataframe is empty, it means that there is something wrong.
        # let's force the break and investigate further
        return True, False, top, top
    if top < 2*len(df):
        # just for good measure.
        new_top = 2*len(df)
    # no problem, return the new top number of rows to query. Let's go!
    return False, False, top, new_top

def iterate_job(ras, decs, gaia_query, tmass_query, path_gaia, path_tmass, columns_tmass_names, TOP=100_000, write=False, timeout=300, start_dec=-999, num_tries=10):
    """
    Iterate through the RAs and Decs to query the Gaia and TMASS databases

    Args:
        ras (array-like): the RAs to iterate through
        decs (array-like): the Decs to iterate through
        gaia_query (str): the query to run on the Gaia database
        tmass_query (str): the query to run on the TMASS database
        path_gaia (str): the path to save the Gaia dataframes
        path_tmass (str): the path to save the TMASS dataframes
        columns_tmass_names (list): the columns to query from the TMASS database
        TOP (int): the top number of rows to query
        write (bool): whether to write the dataframes to disk
        timeout (int): the timeout for the query
        start_dec (float): the starting dec to start the query from for the first ra
        num_tries (int): the number of tries to query the database before giving up
    """
    # keep track of the old top for recording purposes
    OLD_TOP = TOP
    # keep track the force_break flag
    force_break = False
    # loop through the RAs
    for i, (ra0, ra1) in enumerate(zip(ras[:-1], ras[1:])):
        
        if force_break: break
        # create the directory to save the dataframes
        name = f"ra_{ra0:03d}-{ra1:03d}"
        safe_mkdir(join(path_gaia, name))
        safe_mkdir(join(path_tmass, name))
        # create the directory to save the logs
        log_gaia = join(path_gaia, "logs")
        log_tmass = join(path_tmass, "logs")
        safe_mkdir(log_gaia)
        safe_mkdir(log_tmass)
        log_gaia_file = join(log_gaia , f"{name}.txt")
        log_tmass_file = join(log_tmass, f"{name}.txt")
        # loop through the Decs
        time0 = time()
        date0 = datetime.now()
        print_or_write(f"RA: {ra0:03d}-{ra1:03d} {date0}", log_gaia_file, write)
        print_or_write(f"RA: {ra0:03d}-{ra1:03d} {date0}", log_tmass_file, write)
        for j, (dec0, dec1) in enumerate(zip(decs[:-1], decs[1:])):
            t0 = time()
            d0 = datetime.now()
            print_or_write(f"\tDEC: {dec0:02d}-{dec1:02d} {d0}", log_gaia_file, write)
            print_or_write(f"\tDEC: {dec0:02d}-{dec1:02d} {d0}", log_tmass_file, write)
            # keep track of the number of tries
            tries = 0
            # if the starting dec is not -999, then we are continuing from the previous query
            if (ra0 == ras[0]) and (dec0 < start_dec): 
                continue
            if force_break: break
            # loop over the queries until we get a dataframe that is not capped or empty or failed
            while True:
                local_gaia =  f"gaia RA:{ra0:03}-{ra1:03}, DEC:({dec0:02d})-({dec1:02d})"
                local_tmass =  f"tmass RA:{ra0:03}-{ra1:03}, DEC:({dec0:02d})-({dec1:02d})"
                # increment the number of tries
                tries += 1
                # if we have tried too many times, then something is wrong. Let's force the break
                if tries > num_tries:
                    force_break = True
                    print_or_write(f"\n\t{local_gaia} have tried {num_tries} times, something wrong", log_gaia_file, write=write)
                    print_or_write(f"\n\t{local_tmass} have tried {num_tries} times, something wrong", log_tmass_file, write=write)
                    point_in_time(t0, log_gaia_file, write=write)
                    point_in_time(t0, log_tmass_file, write=write)
                if force_break: break
                # sql query to query the Gaia database
                query_gaia = f"""
                {gaia_query}
                WHERE gdr3.ra BETWEEN {ra0} AND {ra1}
                AND gdr3.dec BETWEEN {dec0} AND {dec1}
                """
                # launch the job to query the Gaia database
                df_gaia = launch_job(Gaia.launch_job, query_gaia, duration=timeout)
                # check the result
                force_break, retry, OLD_TOP, TOP = check_df(df_gaia, TOP)
                # if something is wrong, let's force the break
                if force_break: 
                    print_or_write(f"\n\t{local_gaia} is empty", log_gaia_file, write=write)
                    point_in_time(t0, log_gaia_file, write=write)
                    break
                # if failed, let's retry with a larger top
                if retry: 
                    print_or_write(f"\n\t{local_gaia} is capped at {OLD_TOP}, increasing to {TOP}", log_gaia_file, write=write)
                    point_in_time(t0, log_gaia_file, write=write)
                    continue
                # if everything is fine, let's continue the journey
                print_or_write(f"\n\t{local_gaia} has {len(df_gaia)} rows | TOP={TOP}", log_gaia_file, write=write)
                point_in_time(t0, log_gaia_file, write=write)
                # sql query to query the TMASS database
                query_tmass = f"""
                {tmass_query}
                WHERE ra BETWEEN {ra0-0.5} AND {ra1+0.5}
                AND dec BETWEEN {dec0-0.5} AND {dec1+0.5}
                """
                # launch the job to query the TMASS database
                df_tmass = launch_job(tap_tmass.launch_job, query_tmass, cols=columns_tmass_names, duration=timeout)
                # check the result
                force_break, retry, OLD_TOP, TOP = check_df(df_tmass, TOP)
                # if something is wrong, let's force the break
                if force_break: 
                    print_or_write(f"\t{local_tmass} is empty", log_gaia_file, write=write)
                    point_in_time(t0, log_tmass_file, write=write)
                    break
                # if failed, let's retry with a larger top
                if retry: 
                    print_or_write(f"\t{local_tmass} is capped at {OLD_TOP}, increasing to {TOP}", log_gaia_file, write=write)
                    point_in_time(t0, log_tmass_file, write=write)
                    continue
                # if everything is fine, let's continue the journey
                print_or_write(f"\t{local_tmass} has {len(df_tmass)} rows | TOP={TOP}", log_gaia_file, write=write)
                point_in_time(t0, log_tmass_file, write=write)
                # ===========================
                # this is to get the tmass only
                # because we are querying a larger area, we need to filter the result
                df_tmass_only = df_tmass.filter(f"ra > {ra0}").filter(f"ra < {ra1}").filter(f"dec > {dec0}").filter(f"dec < {dec1}")
                df_tmass_only = df_tmass_only.extract()
                # you know the drill
                force_break, retry, OLD_TOP, TOP = check_df(df_tmass_only, TOP)
                if force_break: 
                    print_or_write(f"\tONLY {local_tmass} is empty", log_tmass_file, write=write)
                    point_in_time(t0, log_tmass_file, write=write)
                    break
                if retry: 
                    print_or_write(f"\tONLY {local_tmass} is capped at {OLD_TOP}, increasing to {TOP}", log_tmass_file, write=write)
                    point_in_time(t0, log_tmass_file, write=write)
                    continue
                print_or_write(f"\tONLY {local_tmass} has {len(df_tmass_only)} rows | TOP={TOP}", log_tmass_file, write=write)
                point_in_time(t0, log_tmass_file, write=write)
                # ===========================   
                # join the gaia and tmass
                df_join = df_gaia.join(df_tmass, right_on="designation", left_on="tmass", how="left")
                # delete the columns that are not needed
                df_join.drop(columns=["designation", "tmass"], inplace=True)
                # save the result to the disk
                df_join.export(join(path_gaia, name, f"dec_({dec0})-({dec1}).hdf5"), progress=True)
                print_or_write(f"\tgaia&tmass RA:{ra0:03}-{ra1:03}, DEC:({dec0})-({dec1}) is complete with rows {len(df_join)}", log_gaia_file, write=write)
                point_in_time(t0, log_gaia_file, write=write)
                df_tmass_only.export(join(path_tmass, name, f"dec_({dec0})-({dec1}).hdf5"), progress=True)
                # do the same for the tmass only
                print_or_write(f"\ttmass ONLY RA:{ra0:03}-{ra1:03}, DEC:({dec0})-({dec1}) is complete with rows {len(df_tmass_only)}", log_tmass_file, write=write)
                point_in_time(t0, log_tmass_file, write=write)
                # get out of the while loop
                break
        # after looping for all decs, let's combine them
        df_gaia_tmass = vaex.open_many(glob(join(path_gaia, name, "*.hdf5")))
        # save
        df_gaia_tmass.export(join(path_gaia, f"gaia-{ra0:03d}-{ra1:03d}.hdf5"), progress=True)
        # delete the temporary directory
        check_delete = delete_directory(join(path_gaia, name))
        # check if the directory is deleted
        if not check_delete:
            force_break = True
            write_to(join(path_gaia, f"{name}.txt"), f"Error deleting directory {name}! Stopping loop.")
            point_in_time(time0, log_gaia_file, write=write)
            break
        # tell the world that everything is fine
        print_or_write(f"gaia&tmass RA:{ra0:03}-{ra1:03} is complete with rows {len(df_gaia_tmass)}", log_gaia_file, write=write)
        point_in_time(time0, log_gaia_file, write=write)
        # do the same for the tmass only
        df_tmass_only = vaex.open_many(glob(join(path_tmass, name, "*.hdf5")))
        df_tmass_only.export(join(path_tmass, f"tmass-{ra0:03d}-{ra1:03d}.hdf5"), progress=True)
        check_delete = delete_directory(join(path_tmass, name))
        if not check_delete:
            force_break = True
            write_to(join(path_tmass, f"{name}.txt"), f"Error deleting directory {name}! Stopping loop.")
            point_in_time(time0, log_tmass_file, write=write)
            break
        print_or_write(f"tmass only RA:{ra0:03}-{ra1:03} is complete with rows {len(df_tmass_only)}\n", log_tmass_file, write=write)
        point_in_time(time0, log_tmass_file, write=write)
        # loop again over the next ra

            