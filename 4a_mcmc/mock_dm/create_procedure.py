import sys

tipe = sys.argv[1]
if not tipe in ['z', 'n']:
    raise ValueError('tipe must be z or n')
num = 7 if tipe == 'z' else 6
for model in range(1, 3):
    for data in range(num):
        with open(f"procedure-{tipe}-{model}-{data}.sh", 'w') as f:
            f.write(f"""#!/bin/bash
python generate_init_{model}.py {tipe}
python run_mcmc.py {tipe} {model} {data} True
python plot_chain.py {tipe} {model} {data}
python run_mcmc_again.py {tipe} {model} {data} True
python plot_chain_again.py {tipe} {model} {data}
python plot_corner.py {tipe} {model} {data}
python plot_fit.py {tipe} {model} {data}
python calculate_bic.py {tipe} {model} {data}""")