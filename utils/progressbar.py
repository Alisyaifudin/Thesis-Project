def progressbar(percent=0, width=50, info="", path="", flush=False) -> None:
    """
    Write a progress bar to the console.
    
    Args:
        percent (int): The percentage of the progress bar to fill.
        width (int): The width of the progress bar.
        info (str): The text to display after the progress bar.
        path (str): The path to a file to write the progress bar to.
        flush (bool): Whether to flush the output to the console.
    """
    left = int((width * percent) // 100)
    right = width - left
    
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