model = 2
with open(f"procedure-calculate-probs.sh", 'w') as f:
    f.write(f"#!/bin/bash\n")
    for data in range(13):
        f.write(f"python calculate_probs.py {model} {data}\n")