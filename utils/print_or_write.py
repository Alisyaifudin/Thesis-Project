from .write_to import write_to

def print_or_write(text, path, write=False):
    """
    Print or write text to a file.

    Parameters
    ----------
    text : str
        Text to print or write.
    path : str
        Path to file.
    write : bool, optional
        If True, write text to file. If False, print text to console.
        Default is False.
    """
    if write:
        write_to(path, text)
    else:
        print(text)