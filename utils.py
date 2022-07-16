import signal
from time import time
from requests import HTTPError
from time import sleep

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

def appendName(element, name):
    string = element.split(" AS ")
    if(len(string) == 1):
        return f"{name}.\"{element}\""
    else:
        return f"{name}.\"{string[0]}\" AS {string[1]}"