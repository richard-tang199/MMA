#!/bin/bash

groups=("006" "025" "048" "141" "145" "160" "173")
ratios=("0.02" "0.05" "0.1" "0.15" "0.2")
ratio2=("1" "2" "3" "4")

for group in "${groups[@]}"
do
  for ratio in "${ratios[@]}"
  # simulated
  do
    python train_tsb.py --model_name SAND --mode simulated --anomaly_ratio "$ratio" --data_name "UCR" --group "$group"
    python main.py --model_name PatchContrast --mode simulated --anomaly_ratio "$ratio" --num_epochs 201 --data_name "UCR" --group "$group" --remove_anomaly 0
    python train_other_model.py --model_name DAMP --mode simulated --anomaly_ratio "$ratio" --data_name "UCR" --group "$group"
    python train_other_model.py --model_name MP --mode simulated --anomaly_ratio "$ratio"  --data_name "UCR" --group "$group"
    python train_other_model.py --model_name MAUT --mode simulated --anomaly_ratio "$ratio" --num_epochs 100 --data_name "UCR" --group "$group"
    python train_other_model.py --model_name cad --mode simulated --anomaly_ratio "$ratio" --num_epochs 30 --data_name "UCR" --group "$group"
    python train_other_model.py --model_name mtad_gat --mode simulated --anomaly_ratio "$ratio" --num_epochs 50 --data_name "UCR" --group "$group"
    python train_other_model.py --model_name KMeans --mode simulated --anomaly_ratio "$ratio" --num_epochs 100 --data_name "UCR" --group "$group"
  done
  # rel
  for ratio in "${ratios2[@]}"
  do
    python train_tsb.py --model_name SAND --mode realistic --anomaly_ratio "$ratio" --data_name "UCR" --group "$group"
    python main.py --model_name PatchContrast --mode realistic --anomaly_ratio "$ratio" --num_epochs 201 --data_name "UCR" --group "$group" --remove_anomaly 0
    python main.py --model_name PatchContrast --mode realistic --anomaly_ratio "$ratio" --num_epochs 201 --data_name "UCR" --group "$group" --remove_anomaly 1
    python train_other_model.py --model_name DAMP --mode realistic --anomaly_ratio "$ratio" --data_name "UCR" --group "$group"
    python train_other_model.py --model_name MP --mode realistic --anomaly_ratio "$ratio"  --data_name "UCR" --group "$group"
    python train_other_model.py --model_name MAUT --mode realistic --anomaly_ratio "$ratio" --num_epochs 100 --data_name "UCR" --group "$group"
    python train_other_model.py --model_name cad --mode realistic --anomaly_ratio "$ratio" --num_epochs 30 --data_name "UCR" --group "$group"
    python train_other_model.py --model_name mtad_gat --mode realistic --anomaly_ratio "$ratio" --num_epochs 50 --data_name "UCR" --group "$group"
    python train_other_model.py --model_name "KMeans" --num_epochs 100 --data_name "UCR" --group "$group" --anomaly_ratio "$ratio"
  done
done
