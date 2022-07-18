import signal
from time import time
from requests import HTTPError
from time import sleep
import subprocess
from os.path import join, abspath
from os import pardir, mkdir
import vaex
import numpy as np
from scipy import interpolate

root_data_dir = abspath(join(pardir, "Data"))

# progress bar
def progressbar(percent=0, width=50) -> None:
    left = int((width * percent) // 100)
    right = width - left
    
    tags = "#" * left
    spaces = " " * right
    percents = f"{percent:.0f}%"
    
    print("\r[", tags, spaces, "]", percents, sep="", end="", flush=True)


# add timeout, such that sending request again after some period of time
def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None, minVal=1):

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    t0 = time()
    signal.signal(signal.SIGALRM, handler) 
    signal.alarm(timeout_duration)
    try:
        result = func(*args, **kwargs)
    except TimeoutError as exc:
        result = default
        t1 = time()
        print("too long, requesting again...")
        print(f"time = {round(t1-t0,2)}s")
    except HTTPError:
        result = default
        t1 = time()
        # a litte hacky, need some fixes
        if(t1-t0 < minVal):
            print("service unavailable, sleep for 300s")
            print(f"time = {round(t1-t0,2)}s")
            sleep(300)
            print("continue")
        else:
            print("server not responding, try again")
            print("message", HTTPError)
            print(f"time = {round(t1-t0,2)}s")
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception:
        result = default
        t1 = time()
        print("some error")
        print(Exception)
        print(f"time = {round(t1-t0,2)}s")
    finally:
        signal.alarm(0)
    
    return result

# add AS
def appendName(element, name):
    string = element.split(" AS ")
    if(len(string) == 1):
        return f"{name}.\"{element}\""
    else:
        return f"{name}.\"{string[0]}\" AS {string[1]}"

def runcmd(cmd, verbose = False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass

####################################################################################
####################################################################################
####################################################################################
# need mamajek-spectral-class.hdf5, created at "2. Vertical Number Density/2.3. Survey Completeness.ipynb"
def load_spectral_types():
    file_spectral_class = join(root_data_dir, "mamajek-spectral-class.hdf5")
    df_spectral_class = vaex.open(file_spectral_class)
    df_filtered = df_spectral_class[["SpT", "M_J", "J-H", "H-Ks"]]
    mask = (df_filtered["M_J"] == df_filtered["M_J"])*(df_filtered["H-Ks"] == df_filtered["H-Ks"])*(df_filtered["J-H"] == df_filtered["J-H"])
    df_filtered_good = df_filtered[mask].to_pandas_df()
    df_filtered_good['J-K'] = df_filtered_good['J-H']+df_filtered_good['H-Ks']
    return df_filtered_good

sp = load_spectral_types()
sp_indx = np.array([(not 'O' in s)*(not 'L' in s)\
                        *(not 'T' in s)*(not 'Y' in s)\
                        *(not '.5V' in s) for s in sp['SpT']],
                    dtype='bool')
# Cut out the small part where the color decreases
sp_indx *= (np.roll((sp['J-K']),1)-(sp['J-K'])) <= 0.

main_locus = interpolate.UnivariateSpline((sp['J-K'])[sp_indx],
                                    sp['M_J'][sp_indx],k=3,s=1.)

def main_sequence_cut_r(jk,low=False):
    """Main-sequence cut, based on MJ, high as in low"""
    j_locus= main_locus(jk)
    if low:
        dj = 0.4
    else:
        dj= 0.4-0.1*(j_locus-5.)
        dj*= -1.
    return j_locus+dj
####################################################################################
####################################################################################
####################################################################################