#!/bin/bash
bash run_dm.sh | tee logs/log_dm.txt
bash run_dddm.sh | tee logs/log_dddm.txt
bash run_no.sh | tee logs/log_no.txt