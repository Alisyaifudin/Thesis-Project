#!/bin/bash
cd 1_mock_dm/script/
bash run_process.sh | tee logs/log.txt
cd ../..
cd 2_dm/script/
bash run_process.sh | tee logs/log.txt
bash run_process_comp.sh | tee logs/log_comp.txt
cd ../..
cd 3_dddm/script/
bash run_process.sh | tee logs/log.txt
bash run_process_comp.sh | tee logs/log_comp.txt
cd ../..
cd 4_no/script/
bash run_process.sh | tee logs/log.txt
bash run_process_comp.sh | tee logs/log_comp.txt
cd ../..
cd 5_mond/script/
bash run_process.sh | tee logs/log.txt
bash run_process_comp.sh | tee logs/log_comp.txt