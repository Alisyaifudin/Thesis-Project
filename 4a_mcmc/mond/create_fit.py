model = 2
with open(f"fit-{model}.sh", 'w') as f:
    f.write(f"#!/bin/bash\n")
    for data in range(13):
        f.write(f"python plot_fit.py {model} {data}\n")
        
