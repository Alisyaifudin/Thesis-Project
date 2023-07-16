#!/bin/bash
bash no_dm.sh | tee logs/no_dm.txt
bash no_dddm.sh | tee logs/no_dddm.txt
bash no_no.sh | tee logs/no_no.txt
bash mock_dm.sh | tee logs/mock_dm.txt
bash mock_dddm.sh | tee logs/mock_dddm.txt
bash mock_no.sh | tee logs/mock_no.txt
bash metal_dm.sh | tee logs/metal_dm.txt
bash metal_dddm.sh | tee logs/metal_dddm.txt
bash metal_no.sh | tee logs/metal_no.txt