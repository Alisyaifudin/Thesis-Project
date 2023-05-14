def write_to(path, text, append=True, newline=True):
    """
    Writes and/or append text to a file.

    Args:
        path (str): The path to the file.
        text (str): The text to be written to the file.
        append (bool, optional): Whether to append the text to the file. Defaults to True.
        newline (bool, optional): Whether to add a newline character to the end of the text. Defaults to True.
    """
    mode = "a" if append else "w"
    with open(path, mode) as file:
        file.write(text + "\n" if newline else text)