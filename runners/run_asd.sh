#!/usr/bin/env bash

args=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12")
# window_length analysis
#window_length=("32" "64" "128" "256" "512" "1024")
window_length=("1024")

# ablation analysis
#model_name=("PatchDetector" "PatchAttention" "PatchGru")
model_name=("PatchContrast")

# whether to use dynamic anomaly filtering or not (1 for yes, 0 for no)
remove_args=(1)

for model in "${model_name[@]}"
do
  for g in "${args[@]}"
  do
    for ra in "${remove_args[@]}"
    do
      for w in "${window_length[@]}"
        do
          python main.py --model_name "$model" --num_epochs 101 --window_length "$w" --data_name "ASD" --group "$g" --remove_anomaly "$ra"
          echo "Model: $model, Group: $g, Window Length: $w, Remove Anomaly: $ra"
      done
    done
  done
done