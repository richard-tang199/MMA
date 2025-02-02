#!/usr/bin/env bash

args=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12")

# whether to use contrastive learning or not (1 for yes, 0 for no)
model_name=("PatchContrast" "PatchDetector")

# whether to use dynamic anomaly filtering or not (1 for yes, 0 for no)
remove_args=(0 1)

for model in "${model_name[@]}"
do
  for g in "${args[@]}"
  do
    for ra in "${remove_args[@]}"
    do
      python main.py --model_name "$model" --num_epochs 101 --window_length 1024 --data_name "ASD" --group "$g" --remove_anomaly "$ra"
    done
  done
done