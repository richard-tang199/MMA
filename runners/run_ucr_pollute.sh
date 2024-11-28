#!/bin/bash

groups=("006" "025" "048" "141" "145" "160" "173")
ratios=("0.02" "0.05" "0.1" "0.15")

for group in "${groups[@]}"
do
  python train_tsb.py --model_name SAND --mode realistic --anomaly_ratio 0.02 --data_name "UCR" --group "$group"
  python main.py --model_name PatchContrast --mode realistic --anomaly_ratio 0.02 --num_epochs 201 --data_name "UCR" --group "$group" --remove_anomaly 1
  python train_other_model.py --model_name DAMP --mode realistic --anomaly_ratio 0.02 --data_name "UCR" --group "$group"
  python train_other_model.py --model_name MP --mode realistic --anomaly_ratio 0.02  --data_name "UCR" --group "$group"
  python train_other_model.py --model_name MAUT --mode realistic --anomaly_ratio 0.02 --num_epochs 100 --data_name "UCR" --group "$group"
  python train_other_model.py --model_name cad --mode realistic --anomaly_ratio 0.02 --num_epochs 30 --data_name "UCR" --group "$group"
  python train_other_model.py --model_name mtad_gat --mode realistic --anomaly_ratio 0.02 --num_epochs 50 --data_name "UCR" --group "$group"
  for ratio in "${ratios[@]}"
  do
    python train_tsb.py --model_name SAND --mode simulated --anomaly_ratio "$ratio" --data_name "UCR" --group "$group"
    python main.py --model_name PatchContrast --mode simulated --anomaly_ratio "$ratio" --num_epochs 201 --data_name "UCR" --group "$group" --remove_anomaly 1
    python train_other_model.py --model_name DAMP --mode simulated --anomaly_ratio "$ratio" --data_name "UCR" --group "$group"
    python train_other_model.py --model_name MP --mode simulated --anomaly_ratio "$ratio"  --data_name "UCR" --group "$group"
    python train_other_model.py --model_name MAUT --mode simulated --anomaly_ratio "$ratio" --num_epochs 100 --data_name "UCR" --group "$group"
    python train_other_model.py --model_name cad --mode simulated --anomaly_ratio "$ratio" --num_epochs 30 --data_name "UCR" --group "$group"
    python train_other_model.py --model_name mtad_gat --mode simulated --anomaly_ratio "$ratio" --num_epochs 50 --data_name "UCR" --group "$group"
  done
done
