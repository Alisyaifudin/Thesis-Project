import signal
import pathlib
import numpy as np
import vaex
from astroquery.gaia import Gaia
from astroquery.utils.tap.core import Tap

from datetime import datetime
from os.path import join, abspath
from glob import glob
import sys
from enum import Enum
###################################################################
# import utils from utils.py
current = pathlib.Path(__file__).parent.resolve()
root_dir = join(current, "..", "..")
# get the root of data directory
sys.path.insert(0, root_dir)
from utils import safe_mkdir, delete_directory
root_data_dir = abspath(join(root_dir, "Data"))
###################################################################
# some helper functions

# *******************************************


class TimeoutError(Exception):
    pass


def timeout(func, duration, args=()):
    """
    Run a function with a time limit.

    Args:
        func (function): The function to run.
        duration (int): The maximum number of seconds to allow the function to run.
        args (tuple): The positional arguments to pass to the function. Defaults to an empty tuple.

    Returns:
        dict: A dictionary containing the results of the function call, or an error message if the function timed out or raised an exception.
        The dictionary has two keys:
            - "data": If the function completed successfully, this key maps to the return value of the function.
              If the function failed or timed out, this key maps to None.
            - "error": If the function failed or timed out, this key maps to a dictionary containing information about the error.
              The dictionary has two keys: "type" (a string with the name of the exception) and "message" (a string with the error message).

    Raises:
        TimeoutError: If the function takes longer than `duration` seconds to complete.

    Example usage:
        result = timeout(launch_job, 10, args=(arg1, arg2))
        if result["error"]:
            print("Error:", result["error"]["type"], result["error"]["message"])
        else:
            print("Result:", result["data"])
    """
    def handler(signum, frame):
        raise TimeoutError("Function timed out")

    # Set the signal handler for SIGALRM
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(duration)

    try:
        result = func(*args)
        data = result
        error = None
    except TimeoutError:
        data = None
        error = {"type": "TimeoutError", "message": "Function timed out"}
    except Exception as e:
        data = None
        error = {"type": type(e).__name__, "message": str(e)}

    # Disable the alarm
    signal.alarm(0)

    return {"data": data, "error": error}
# *******************************************


def launch_job(job_func, query, cols="", duration=10):
    """
    Launch a job with a timeout.

    Args:
        job_func (func): The job to launch.
        query (str): The query to run.
        cols (str): Rename the cols name if needed.
        duration (int): The timeout duration.

    Returns:
        df (vaex.dataframe): The dataframe or None.
    """
    # run the job and wrap in timeout
    job = timeout(job_func, args=(query,), duration=duration)
    res = {"data": None, "error": None}
    # print error if job failed
    if job['data'] == None:
        res["error"] = job['error']
    # convert the result into vaex.dataframe if successful
    else:
        result = job['data'].get_results()
        df = result.to_pandas()
        if cols != "":
            df.columns = cols
        df = vaex.from_pandas(df)
        res["data"] = df
    return res
# *******************************************


def append_name(element, name):
    """
    Append a table name to a column name.

    Parameters:
    ----------
    element: str 
        The column name.
    name: str 
        The table name.

    Returns:
    -------
        str: The column name with the table name appended.

    Example usage:
    --------------
        [in]:  append_name("id", "users")

        [out]: users.\"id\"

        [in]:  append_name("id AS user_id", "users")

        [out]: users.\"id\" AS user_id
    """
    string = element.split(" AS ")
    if (len(string) == 1):
        return f"{name}.\"{element}\""
    else:
        return f"{name}.\"{string[0]}\" AS {string[1]}"

def progress_bar(current, total, interval=None, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))

    sys.stdout.write(f'\rProgress: [{arrow}{spaces}] {percent:.01f} % | {interval}')
    sys.stdout.flush()
# *******************************************
###################################################################
# Create a directory for Gaia DR3 and 2MASS data or if it already exists, just move on
# Create a directory for Gaia DR3 and 2MASS data
# or if it already exists, just move on
ra_name = "Gaia-2MASS"
gaia_data_dir = join(root_data_dir, ra_name)
safe_mkdir(gaia_data_dir)
# Do the same for 2MASS data
ra_name = "2MASS"
tmass_data_dir = join(root_data_dir, ra_name)
safe_mkdir(tmass_data_dir)
###################################################################
# initializations
column_gaia = [
    "source_id", "ra", "dec",
    "pm", "pmra", "pmra_error AS e_pmra", "pmdec", "pmdec_error AS e_pmdec",
    "parallax", "parallax_error AS e_parallax",
    "phot_g_mean_mag AS g",	"phot_bp_mean_mag AS bp", "phot_rp_mean_mag AS rp",
    "phot_bp_mean_flux_over_error AS fb_over_err",
    "phot_rp_mean_flux_over_error AS fr_over_err",
    "ruwe",
    "phot_bp_rp_excess_factor AS excess_factor",
    "radial_velocity AS rv_gaia", "radial_velocity_error AS e_rv_gaia",
    "l", "b",
]


column_join_table = ["original_ext_source_id AS tmass_id"]

gaia_alias = "gdr3"
join_table_alias = "join_table"

column_gaia = list(map(lambda x: append_name(x, gaia_alias), column_gaia))
column_join_table = list(
    map(lambda x: append_name(x, join_table_alias), column_join_table))

column_gaia = column_gaia + column_join_table

# 2MASS tap endpoint
tap_tmass = Tap(url="https://irsa.ipac.caltech.edu/TAP/sync")

columns_tmass = ["ra", "dec", "j_m", "k_m",
                 "designation", "ph_qual", "use_src", "rd_flg"]

# rename the table columns as
columns_tmass_names = ["ra", "dec", "j", "k",
                       "designation", "ph_qual", "use_src", "rd_flg"]

tmass_table = "fp_psc"
column_tmass = list(map(lambda x: append_name(x, tmass_table), columns_tmass))
###################################################################
# download functions


class Next(Enum):
    BREAK = 1
    RETRY = 2
    CONTINUE = 3


def get_df(options):
    limit = options.get("limit", {
        "gaia": 1000,
        "tmass": 1000
    })
    time_out = options.get("time_out", 120)
    result = {
        "join": None,
        "tmass": None,
        "gaia": None,
        "next": None,
        "info": None,
        "limit": limit
    }    
    
    ra_low = options.get("ra_low")
    ra_high = options.get("ra_high")
    dec_low = options.get("dec_low")
    dec_high = options.get("dec_high")

    query_gaia = f"""
    SELECT TOP {limit["gaia"]} {', '.join(column_gaia)}
    FROM gaiadr3.gaia_source AS {gaia_alias}
    RIGHT JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS {join_table_alias} ON {join_table_alias}.source_id = {gaia_alias}.source_id
    WHERE {gaia_alias}.ra BETWEEN {ra_low} AND {ra_high}
    AND {gaia_alias}.dec BETWEEN {dec_low} AND {dec_high}
    AND {join_table_alias}.original_ext_source_id IS NOT NULL
    """

    query_tmass = f"""
    SELECT TOP {limit["tmass"]} {", ".join(columns_tmass)} 
    FROM {tmass_table}
    WHERE ra BETWEEN {ra_low-0.3} AND {ra_high+0.3}
    AND dec BETWEEN {dec_low-0.3} AND {dec_high+0.3}
    """

    job_gaia = launch_job(Gaia.launch_job, query_gaia, duration=time_out)
    df_gaia = None if job_gaia["error"] else job_gaia["data"]
    df_join = None
    df_tmass = None

    if job_gaia["error"]:
        if job_gaia["error"]["type"] in ["TimeoutError", "IncompleteRead", "ConnectionResetError"]:
            result["next"] = Next.RETRY
            result["info"] = f"\tGaia job error, with message: {job_gaia['error']}\nTry again..."
        else:
            result["next"] = Next.BREAK
            result["info"] = f"Gaia job error, with message: {job_gaia['error']}"
        return result
    if len(df_gaia) == limit["gaia"]:
        result["next"] = Next.RETRY
        result["info"] = f"Gaia job capped at {limit['gaia']}, increase to {limit['gaia']*2}"
        result["limit"]['gaia'] = 2 * limit["gaia"]
        return result
    job_tmass = launch_job(tap_tmass.launch_job, query_tmass, cols=columns_tmass_names, duration=time_out)
    df_tmass = None if job_tmass["error"] else job_tmass["data"]
 
    if job_tmass["error"]:
        if job_tmass["error"]["type"] in ["TimeoutError", "IncompleteRead", "ConnectionResetError", "HTTPError", "E19"]:
            result["next"] = Next.RETRY
            result["info"] = f"\t2MASS job error, with message: {job_tmass['error']}\nTry again..."
        else:
            result["next"] = Next.BREAK
            result["info"] = f"2MASS job error, with message: {job_tmass['error']}"
        return result
    if len(df_tmass) == limit['tmass']:
        result["next"] = Next.RETRY
        result["info"] = f"2MASS job capped at {limit['tmass']}, increase to {limit['tmass']*2}"
        result["limit"]['tmass'] = 2 * limit['tmass']
        return result
    df_join = df_gaia.join(df_tmass, right_on="designation", left_on="tmass_id", how="inner", rsuffix="_tmass")
    df_tmass = df_tmass.filter(f"ra > {ra_low}").filter(f"ra < {ra_high}").filter(f"dec > {dec_low}").filter(f"dec < {dec_high}")
    df_tmass = df_tmass.extract()
    # convert to pandas dataframe, because vaex has a weird bug
    # when len(df_tmass) is 2^x, x is an integer
    # I know it's annoying, but it's a temporary solution
    df_tmass = df_tmass.to_pandas_df()
    # convert back to vaex dataframe
    df_tmass = vaex.from_pandas(df_tmass)
    df_join.drop(["ra_tmass", "dec_tmass", "designation"], inplace=True)
    df_pandas = df_join.to_pandas_df()
    # drop tmass_id duplicates
    df_pandas.drop_duplicates(subset=["tmass_id"], inplace=True)
    result["join"] = vaex.from_pandas(df_pandas)
    result["gaia"] = df_gaia
    result["tmass"] = df_tmass
    result["info"] = "Gaia and 2MASS job success"
    result["next"] = Next.CONTINUE
    return result


def iterate_job(options):
    limit = options.get("limit", {
        "gaia": 100_000,
        "tmass": 100_000
    })
    # keep track force_break flags
    force_break = False
    # read options
    ras = options.get("ras")
    decs = options.get("decs")
    dec_start = options.get("dec_start", -90)
    num_tries = options.get("num_tries", 10)
    time_out = options.get("time_out", 120)
    dir_path = options.get("dir_path")
    n = options.get("n", 21)
    # loop through the RAs
    for i, (ra0, ra1) in enumerate(zip(ras[:-1], ras[1:])):
        if force_break: break
        # create the directory to save the dataframes
        ra_name = f"ra_{ra0:03d}-{ra1:03d}"
        safe_mkdir(join(dir_path["gaia"], ra_name))
        safe_mkdir(join(dir_path["tmass"], ra_name))
        date0 = datetime.now()
        print("================================")
        print(f"RA: {ra0:03d}-{ra1:03d} | {date0} | limit: {limit}")
        for j, (dec0, dec1) in enumerate(zip(decs[:-1], decs[1:])):
            if force_break: break
            if (ra0 == ras[0]) and (dec0 < dec_start): 
                continue
            d0 = datetime.now()
            dec_name = f"dec_({dec0:02d})_({dec1:02d})"
            print(f"\tRA: {ra_name}; DEC: {dec_name} | {d0}")
            # keep track of the number of tries
            tries = 0
            result = None
            try_log =  f"RA:{ra0:03}-{ra1:03}, DEC:({dec0:02d})-({dec1:02d})"
            # if we have tried too many times, then something is wrong. Let's force the break
            if tries > num_tries:
                force_break = True
                print(f"\n\tGaia {try_log} have tried {num_tries} times, something wrong")
                break
            decs_inner = np.linspace(dec0, dec1, n)
            result = {
                "join": None,
                "tmass": None,
                "gaia": None,
                "next": None,
                "info": None,
            }
            progress_bar(0, n-1, interval=d0-d0)
            for i, (dec_low, dec_high) in enumerate(zip(decs_inner[:-1], decs_inner[1:])):
                if force_break: break
                opts = {
                    "limit": limit,
                    "time_out": time_out,
                    "ra_low": ra0,
                    "ra_high":  ra1,
                    "dec_low": dec_low,
                    "dec_high": dec_high
                }
                while True:
                    res = get_df(opts)
                    if res["next"] == Next.BREAK:
                        force_break = True
                        result = res
                        print(result["info"])
                        break
                    elif res["next"] == Next.RETRY:
                        limit = res["limit"]
                        print(f"\tGaia {try_log} retrying {tries+1} times\n\t{res['info']}")
                        tries += 1
                        continue
                    elif res["next"] == Next.CONTINUE:
                        progress_bar(i+1, n-1, interval=datetime.now()-d0)
                        if i == 0:
                            result = res
                        else:
                            result["join"] = result["join"].concat(res["join"])
                            result["tmass"] = result["tmass"].concat(res["tmass"])
                            result["gaia"] = result["gaia"].concat(res["gaia"])
                            result["info"] = res["info"]
                        break
            if force_break: break
            print("\n\t", result["info"])
            print(f"\tjoin: {len(result['join'])}\n\ttmass: {len(result['tmass'])}\n\tgaia: {len(result['gaia'])}")
            df_tmass = result["tmass"]
            df_join = result["join"]
            
            df_tmass.export(join(dir_path["tmass"], ra_name, f"{dec_name}.hdf5"))
            df_join.export(join(dir_path["gaia"], ra_name, f"{dec_name}.hdf5"))
            d1 = datetime.now()
            print("\tduration: ", d1 - d0)
        # after looping for all decs, let's combine them
        if force_break: break
        print("Combine all ra")
        df_join = vaex.open_many(glob(join(dir_path["gaia"], ra_name, f"*.hdf5")))
        df_tmass = vaex.open_many(glob(join(dir_path["tmass"], ra_name, f"*.hdf5")))

        df_join.export(join(dir_path["gaia"], f"gaia-{ra0:03d}-{ra1:03d}.hdf5"))
        df_tmass.export(join(dir_path["tmass"], f"tmass-{ra0:03d}-{ra1:03d}.hdf5"))

        delete_join_ra_dir = delete_directory(join(dir_path["gaia"], ra_name))
        delete_tmass_ra_dir = delete_directory(join(dir_path["tmass"], ra_name))
        if not delete_join_ra_dir:
            force_break = True
            print(join(dir_path["gaia"], f"{ra_name}.txt"), f"Error deleting directory {ra_name}! Stopping loop.")
            break
        if not delete_tmass_ra_dir:
            force_break = True
            print(join(dir_path["tmass"], f"{ra_name}.txt"), f"Error deleting directory {ra_name}! Stopping loop.")
            break
        date1 = datetime.now()
        print("Complete RA: ", ra_name)
        print("Total duration", date1 - date0)
        print("***************************")

###################################################################
# main program
if __name__ == "__main__":
    try:

        if len(sys.argv) < 5:
            print(
                "Usage: python gaia-tmass.py ra_low ra_high dec_low dec_high [dec_start]")
            sys.exit(1)
        ra_low = int(sys.argv[1])
        ra_high = int(sys.argv[2])
        dec_low = int(sys.argv[3])
        dec_high = int(sys.argv[4])
        dec_start = -90.
        if len(sys.argv) == 6:
            dec_start = int(sys.argv[5])
        # create the directory to save the dataframes
        dir_path = {
            "gaia": gaia_data_dir,
            "tmass": tmass_data_dir
        }
        # limit
        limit = {
            "gaia": 100_000,
            "tmass": 100_000
        }
        # ra and dec ranges
        ras = np.arange(ra_low, ra_high+0.1, 1).astype(int)
        decs = np.arange(dec_low, dec_high+0.1, 1).astype(int)
        # options
        options = {
            "ras": ras,
            "decs": decs,
            "dir_path": dir_path,
            "limit": limit,
            "num_tries": 25,
            "dec_start": dec_start,
            "time_out": 10*60,
            "n": 6
        }
        # run the main program
        print("ras", ras)
        print("decs", decs)
        iterate_job(options)
    except Exception as e:
        print(e)
        sys.exit(1)
