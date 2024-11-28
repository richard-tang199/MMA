#!/usr/bin/env bash

group_list=("real_satellite_data_1" "real_satellite_data_2")
for group in "${group_list[@]}"
do
  python main.py --model_name PatchContrast --data_name "sate" --group "$group" --remove_anomaly 1 --window_length 1024
done
