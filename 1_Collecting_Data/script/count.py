# This script is used to count the number of stars in each 
# 1x1 square of ra and dec in 2MASS catalog.
import numpy as np
import vaex
from astroquery.utils.tap.core import Tap
# utils
import signal

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
    res = { "data": None, "error": None }
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

# main program
if __name__ == "__main__":
    tap_tmass = Tap(url="https://irsa.ipac.caltech.edu/TAP/sync")
    tmass_table = "fp_psc"

    ras = np.arange(0, 360+0.1, 1).astype(int)
    decs = np.arange(-90, 90+0.1, 1).astype(int)
    for (ra_low, ra_high) in zip(ras[:-1], ras[1:]):
        print("=====================================")
        for (dec_low, dec_high) in zip(decs[:-1], decs[1:]):
            query_count = f"""
            SELECT COUNT(*)
            FROM {tmass_table}
            WHERE ra BETWEEN {ra_low} AND {ra_high}
            AND dec BETWEEN {dec_low} AND {dec_high}
            """
            job_count = launch_job(tap_tmass.launch_job, query_count, cols=["count"], duration=60)
            df_count = None if job_count["error"] else job_count["data"]
            if job_count["error"]:
                print(job_count["error"], f"\nra = [{ra_low}, {ra_high}], dec = [{dec_low}, {dec_high}]")
            else:
                print(f"ra = [{ra_low}, {ra_high}], dec = [{dec_low}, {dec_high}], count = {df_count['count'].to_numpy()[0]}")