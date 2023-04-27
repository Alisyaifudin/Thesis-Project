from glob import glob
import sys

tipe = sys.argv[1]
if not tipe in ['z', 'n']:
    raise ValueError('tipe must be z or n')
files = glob(f"procedure-{tipe}-*.sh")
files.sort()
with open(f"run_procedure-{tipe}.sh", 'w') as f:
    f.write("#!/bin/bash\n")
    for i, file in enumerate(files):
        f.write(f"echo {file}\n")
        f.write(f"bash {file}\n")