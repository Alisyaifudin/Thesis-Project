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
        print("Function timed out")
        data = None
        error = {"type": "TimeoutError", "message": "Function timed out"}
    except Exception as e:
        data = None
        error = {"type": type(e).__name__, "message": str(e)}

    # Disable the alarm
    signal.alarm(0)

    return {"data": data, "error": error}
