#!/bin/bash
bash run_process_z.sh | tee logs/log_z.txt
bash run_process_n.sh | tee logs/log_n.txt
bash run_process_dm.sh | tee logs/log_dm.txt
bash run_process_dm_comp.sh | tee logs/log_dm_comp.txt
bash run_process_dddm.sh | tee logs/log_dddm.txt
bash run_process_dddm_comp.sh | tee logs/log_dddm_comp.txt
bash run_process_no.sh | tee logs/log_no.txt
bash run_process_no_comp.sh | tee logs/log_no_comp.txt