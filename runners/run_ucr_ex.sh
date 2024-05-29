#!/usr/bin/env bash

groups=("006" "025" "048" "141" "145" "160" "173")

for g in "${groups[@]}"
do
  python explain.py --model_name PatchContrast --data_name UCR --group "$g"
  python explain.py --model_name mtad_gat --data_name UCR --group "$g"
  python explain.py --model_name TranAD --data_name UCR --group "$g"
  python explain.py --model_name MAUT --data_name UCR --group "$g"
  python explain.py --model_name cad --data_name UCR --group "$g"
done
