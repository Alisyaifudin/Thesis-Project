#!/usr/bin/env bash
python mcmc-DD-DM.py F 5000 100000 DD &
python mcmc-DM.py F 5000 100000 DM &
python mcmc-no.py F 5000 100000 no &
python mcmc-DD-DM.py G 5000 100000 DD &
python mcmc-DM.py G 5000 100000 DM &
python mcmc-no.py G 5000 100000 no