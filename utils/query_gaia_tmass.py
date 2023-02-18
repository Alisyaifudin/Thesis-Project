from utils import print_or_write, safe_mkdir, launch_job, write_to, delete_directory
from os.path import join
import vaex
from astroquery.gaia import Gaia
from astroquery.utils.tap.core import Tap
from glob import glob

tap_tmass = Tap(url="https://irsa.ipac.caltech.edu/TAP/sync")

def check_df(df, top):
    """
    Check if the dataframe is None, capped, or empty

    Args:
        df (DataFrame-like): the dataframe to check
        top (int): the top number of rows to query
    
    Returns:
        (bool, bool, int): (force_break, retry, top, new_top)
    """
    new_top = top
    if df is None:
        return False, True, top, top
    elif len(df) == top:
        new_top = 2*top
        return False, True, top, new_top
    elif len(df) == 0:
        return True, False, top, top
    if top > 2*len(df):
        new_top = 2*len(df)
    return False, False, top, new_top

def iterate_job(ras, decs, gaia_query, tmass_query, path_gaia, path_tmass, columns_tmass_names, TOP=100_000, write=False, timeout=300, start_dec=-999):
    """
    Iterate through the RAs and Decs to query the Gaia and TMASS databases

    Args:
        ras (array-like): the RAs to iterate through
        decs (array-like): the Decs to iterate through
        gaia_query (str): the query to run on the Gaia database
        tmass_query (str): the query to run on the TMASS database
        TOP (int): the top number of rows to query
    """
    OLD_TOP = TOP
    force_break = False
    for i, (ra0, ra1) in enumerate(zip(ras[:-1], ras[1:])):
        if force_break: break
        name = f"ra_{ra0:03d}-{ra1:03d}"
        safe_mkdir(join(path_gaia, name))
        safe_mkdir(join(path_tmass, name))
        log_gaia = join(path_gaia, "logs")
        log_tmass = join(path_tmass, "logs")
        safe_mkdir(log_gaia)
        safe_mkdir(log_tmass)
        log_gaia_file = join(log_gaia , f"{name}.txt")
        log_tmass_file = join(log_tmass, f"{name}.txt")
        for j, (dec0, dec1) in enumerate(zip(decs[:-1], decs[1:])):
            if (ra0 == ras[0]) and (dec0 < start_dec): 
                continue
            if force_break: break
            while True:
                if force_break: break
                query_gaia = f"""
                {gaia_query}
                WHERE gdr3.ra BETWEEN {ra0} AND {ra1}
                AND gdr3.dec BETWEEN {dec0} AND {dec1}
                """
                df_gaia = launch_job(Gaia.launch_job, query_gaia, duration=timeout)
                force_break, retry, OLD_TOP, TOP = check_df(df_gaia, TOP)
                if force_break: 
                    print_or_write(f"\n\tgaia RA:{ra0:03}-{ra1:03}, DEC:({dec0:02d})-({dec1:02d}) is empty", log_gaia_file, write=write)
                    break
                if retry: 
                    print_or_write(f"\n\tgaia RA:{ra0:03}-{ra1:03}, DEC:({dec0:02d})-({dec1:02d}) is capped", log_gaia_file, write=write)
                    continue
                print_or_write(f"\n\tgaia RA:{ra0:03}-{ra1:03}, DEC:({dec0:02d})-({dec1:02d}) has {len(df_gaia)} rows", log_gaia_file, write=write)
                query_tmass = f"""
                {tmass_query}
                WHERE ra BETWEEN {ra0-0.5} AND {ra1+0.5}
                AND dec BETWEEN {dec0-0.5} AND {dec1+0.5}
                """
                df_tmass = launch_job(tap_tmass.launch_job, query_tmass, cols=columns_tmass_names, duration=timeout)
                force_break, retry, OLD_TOP, TOP = check_df(df_tmass, TOP)
                if force_break: 
                    print_or_write(f"\ttmass RA:{ra0:03}-{ra1:03}, DEC:({dec0:02d}]-[{dec1:02d}] is empty", log_gaia_file, write=write)
                    break
                if retry: 
                    print_or_write(f"\ttmass RA:{ra0:03}-{ra1:03}, DEC:({dec0:02d})-({dec1:02d}) is capped at {OLD_TOP}, increasing to {TOP}", log_gaia_file, write=write)
                    continue
                print_or_write(f"\ttmass RA:{ra0:03}-{ra1:03}, DEC:({dec0:02d})-({dec1:02d}) has {len(df_tmass)} rows", log_gaia_file, write=write)
                # ===========================
                # this is to get the tmass only
                df_tmass_only = df_tmass.filter(f"ra > {ra0}").filter(f"ra < {ra1}").filter(f"dec > {dec0}").filter(f"dec < {dec1}")
                df_tmass_only = df_tmass_only.extract()
                force_break, retry, OLD_TOP, TOP = check_df(df_tmass_only, TOP)
                if force_break: 
                    print_or_write(f"\ttmass ONLY RA:{ra0:03}-{ra1:03}, DEC:({dec0:02d})-({dec1:02d}) is empty", log_tmass_file, write=write)
                    break
                if retry: 
                    print_or_write(f"\ttmass ONLY RA:{ra0:03}-{ra1:03}, DEC:({dec0:02d})-({dec1:02d}) is capped at {OLD_TOP}, increasing to {TOP}", log_tmass_file, write=write)
                    continue
                print_or_write(f"\ttmass ONLY RA:{ra0:03}-{ra1:03}, DEC:({dec0:02d})-({dec1:02d}) has {len(df_tmass)} rows", log_tmass_file, write=write)
                # ===========================   
                df_join = df_gaia.join(df_tmass, right_on="designation", left_on="tmass", how="left")
                df_join.drop(columns=["designation", "tmass"], inplace=True)
                df_join.export(join(path_gaia, name, f"dec_({dec0:02d})-({dec1:02d}).hdf5"), progress=True)
                print_or_write(f"\tgaia&tmass RA:{ra0:03}-{ra1:03}, DEC:({dec0:02d})-({dec1:02d}) is complete with rows {len(df_join)}", log_gaia_file, write=write)
                df_tmass_only.export(join(path_tmass, name, f"dec_({dec0:02d})-({dec1:02d}).hdf5"), progress=True)
                print_or_write(f"\ttmass ONLY RA:{ra0:03}-{ra1:03}, DEC:({dec0:02d})-({dec1:02d}) is complete with rows {len(df_tmass_only)}", log_tmass_file, write=write)
                break
        df_gaia_tmass = vaex.open_many(glob(join(path_gaia, name, "*.hdf5")))
        df_gaia_tmass.export(join(path_gaia, f"gaia-{ra0:03d}-{ra1:03d}.hdf5"), progress=True)
        check_delete = delete_directory(join(path_gaia, name))
        if not check_delete:
            force_break = True
            write_to(join(path_gaia, f"{name}.txt"), f"Error deleting directory {name}! Stopping loop.")
            break
        print_or_write(f"gaia&tmass RA:{ra0:03}-{ra1:03} is complete with rows {len(df_gaia_tmass)}", log_gaia_file, write=write)
        df_tmass_only = vaex.open_many(glob(join(path_tmass, name, "*.hdf5")))
        df_tmass_only.export(join(path_tmass, f"tmass-{ra0:03d}-{ra1:03d}.hdf5"), progress=True)
        check_delete = delete_directory(join(path_tmass, name))
        if not check_delete:
            force_break = True
            write_to(join(path_tmass, f"{name}.txt"), f"Error deleting directory {name}! Stopping loop.")
            break
        print_or_write(f"tmass only RA:{ra0:03}-{ra1:03} is complete with rows {len(df_tmass_only)}\n", log_tmass_file, write=write)