from .timeout import timeout
import vaex

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
    df = None
    # print error if job failed
    if job['data'] == None:
        print(job['error'])
    # convert the result into vaex.dataframe if successful
    else:
        result = job['data'].get_results()
        df = result.to_pandas()
        if cols != "":
            df.columns = cols
        df = vaex.from_pandas(df)
    return df