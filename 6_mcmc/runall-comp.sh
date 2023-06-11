#!/bin/bash
cd 2_dm/script/
bash run_process_comp.sh | tee logs/log_comp-2.txt
cd ../..
cd 3_dddm/script/
bash run_process_comp.sh | tee logs/log_comp-2.txt
cd ../..
cd 4_no/script/
bash run_process_comp.sh | tee logs/log_comp-2.txt
cd ../..
cd 5_mond/script/
bash run_process_comp.sh | tee logs/log_comp-2.txt