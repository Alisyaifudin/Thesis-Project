from .write_to import write_to

def print_or_write(text, path, write=False):
    if write:
        write_to(path, text)
    else:
        print(text)