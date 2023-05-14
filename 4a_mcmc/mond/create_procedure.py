for model in range(1, 3):
    for data in range(13):
        with open(f"procedure-{model}-{data}.sh", 'w') as f:
            f.write(f"""#!/bin/bash
python generate_init_{model}.py
python run_mcmc.py {model} {data} True
python plot_chain.py {model} {data}
python run_mcmc_again.py {model} {data} True
python plot_chain_again.py {model} {data}
python plot_corner.py {model} {data}
python plot_fit.py {model} {data}
python calculate_bic.py {model} {data}
python plot_a0.py {model} {data}""")