#!/bin/bash
bash mock_dm.sh | tee logs/mock_dm.txt
bash mock_dddm.sh | tee logs/mock_dddm.txt
bash mock_no.sh | tee logs/mock_no.txt