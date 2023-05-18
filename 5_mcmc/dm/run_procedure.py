from glob import glob

files = glob("procedure-*.sh")
files.sort()
with open(f"run_procedure.sh", 'w') as f:
    f.write("#!/bin/bash\n")
    for i, file in enumerate(files):
        f.write(f"echo {file}\n")
        f.write(f"bash {file}\n")