#!/usr/bin/env bash

# ablation analysis
#model_name=("PatchDetector" "PatchAttention" "PatchGru")
model_name=("PatchContrast")

# window_length analysis
#window_length=("32" "64" "128" "256" "512" "1024")
window_length=("1024")

group_list=("real_satellite_data_1" "real_satellite_data_2" "real_satellite_data_3")
remove_args=(1)

for model in "${model_name[@]}"
do
  for group in "${group_list[@]}"
  do
    for ra in "${remove_args[@]}"
    do
      for window_length in "${window_length[@]}"
        do
          echo "Running $model on $group with window_length=$window_length and remove_anomaly=$ra"
          python main.py --model_name "$model" --data_name "sate" --group "$group" --remove_anomaly "$ra" --window_length "$window_length"
      done
    done
  done
done