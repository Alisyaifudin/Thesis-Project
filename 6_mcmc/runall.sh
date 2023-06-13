#!/bin/bash
cd 3_dddm/script/
bash run_process_no.sh | tee logs/log_no.txt
bash run_process_comp.sh | tee logs/log_comp.txt
cd ../..
cd 4_no/script/
bash run_process_no.sh | tee logs/log_no.txt
bash run_process_comp.sh | tee logs/log_comp.txt