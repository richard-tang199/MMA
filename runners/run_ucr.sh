#!/usr/bin/env bash

for group in {1..250}
do
  python main.py --model_name "PatchContrast" --num_epochs 201 --data_name "UCR" --group "$group" --remove_anomaly "1"
done