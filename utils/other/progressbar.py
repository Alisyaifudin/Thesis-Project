def progressbar(percent=0, width=50, info="", path="", flush=False, delta=0) -> None:
    """
    Write a progress bar to the console.
    
    Args:
        percent (int): The percentage of the progress bar to fill.
        width (int): The width of the progress bar.
        info (str): The text to display after the progress bar.
        path (str): The path to a file to write the progress bar to.
        flush (bool): Whether to flush the output to the console.
        delta (int): The time in seconds since the start of the progress bar.
    """
    h, m, s = 0, 0, 0
    ht, mt, st = 0, 0, 0
    est = None
    if delta != 0:
        est = delta*(100/percent-1)
        h = int(est // 3600)
        m = int((est % 3600) // 60)
        s = int(est % 60)
        ht = int(delta // 3600)
        mt = int((delta % 3600) // 60)
        st = int(delta % 60)
    delta = f"{ht:02d}:{mt:02d}:{st:02d}"
    est = f"{h:02d}:{m:02d}:{s:02d}" if est is not None else None
    left = int((width * percent) // 100)
    right = width - left
    info = f"<{delta}|{est}> {info}" if est is not None else info
    tags = "#" * left
    spaces = " " * right
    percents = f"{percent:.0f}%"
    text = f"\r[{tags}{spaces}] {percents} {info}"
    if(flush):
        print(text, end="", flush=True)
    else:
        print(text)
    if(path != ""):
        with open(path, 'a') as f:
            f.write(f"{text}")