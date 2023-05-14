import numpy as np
ras = np.arange(0, 360.1, 30).astype(int)
for ra0, ra1 in zip(ras[:-1], ras[1:]):
    with open(f"procedures/procedure-{ra0:03d}-{ra1:03d}.sh", 'w') as f:
        f.write(f"""#!/bin/bash\npython -u gaia-tmass.py {ra0} {ra1} -90 90 | tee log/log-{ra0:03d}-{ra1:03d}.txt
        """)
