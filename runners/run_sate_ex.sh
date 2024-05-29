#!/usr/bin/env bash

groups=("real_satellite_data_1" "real_satellite_data_2")

for g in "${groups[@]}"
do
  python explain.py --model_name PatchContrast --data_name sate --group "$g"
  python explain.py --model_name mtad_gat --data_name sate --group "$g"
  python explain.py --model_name TranAD --data_name sate --group "$g"
  python explain.py --model_name MAUT --data_name sate --group "$g"
  python explain.py --model_name cad --data_name sate --group "$g"
done

