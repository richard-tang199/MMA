#!/usr/bin/env bash

groups=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12")

for g in "${groups[@]}"
do
  python explain.py --model_name PatchContrast --data_name ASD --group "$g"
  python explain.py --model_name mtad_gat --data_name ASD --group "$g"
  python explain.py --model_name TranAD --data_name ASD --group "$g"
  python explain.py --model_name MAUT --data_name ASD --group "$g"
  python explain.py --model_name cad --data_name ASD --group "$g"
done

