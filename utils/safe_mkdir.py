from os import mkdir

def safe_mkdir(path: str):
    try:
        mkdir(path)
        print(f"Creating {path} dir in Data dir")
    except FileExistsError:
        print(f"Directory {path} already exist. Good to go!")