from os import mkdir

def safe_mkdir(path: str):
    """
    Create a directory if it doesn't exist

    Args:
        path (str): Path to the directory to create
    """
    try:
        mkdir(path)
        print(f"Creating {path} dir in Data dir")
    except FileExistsError:
        print(f"Directory {path} already exist. Good to go!")