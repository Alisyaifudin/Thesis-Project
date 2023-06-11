#!/bin/bash
cd 2_dm/script/
bash run_process_no.sh | tee logs/log_no.txt
bash run_process_comp.sh | tee logs/log_comp.txt
cd ../..
cd 3_dddm/script/
bash run_process_no.sh | tee logs/log_no.txt
bash run_process_comp.sh | tee logs/log_comp.txt
cd ../..
cd 4_no/script/
bash run_process_no.sh | tee logs/log_no.txt
bash run_process_comp.sh | tee logs/log_comp.txt
cd ../..
cd 5_mond/script/
bash run_process_no.sh | tee logs/log_no.txt
bash run_process_comp.sh | tee logs/log_comp.txt