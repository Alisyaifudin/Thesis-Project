from glob import glob
for index in range(1,3):
    files = glob(f"procedure-{index}*.sh")
    files.sort()
    with open(f"run_procedure_{index}.sh", 'w') as f:
        f.write("#!/bin/bash\n")
        for i, file in enumerate(files):
            f.write(f"echo {file}\n")
            f.write(f"bash {file}\n")