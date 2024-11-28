#!/bin/bash

groups=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12")
ratios=("0.02" "0.05" "0.1" "0.15")

for group in "${groups[@]}"
do
  # realistic anomaly pollution case
  python main.py --model_name PatchContrast --mode realistic --anomaly_ratio 0.02 --num_epochs 101 --data_name "ASD" --group "$group" --remove_anomaly 1
  python train_other_model.py --model_name mtad_gat --mode realistic --anomaly_ratio 0.02 --num_epochs 50 --data_name "ASD" --group "$group"
  python train_other_model.py --model_name MAUT --mode realistic --anomaly_ratio 0.02 --num_epochs 100 --data_name "ASD" --group "$group"
  python train_other_model.py --model_name usad --mode realistic --anomaly_ratio 0.02 --num_epochs 100 --data_name "ASD" --group "$group"
  python train_other_model.py --model_name cad --mode realistic --anomaly_ratio 0.02 --num_epochs 30 --data_name "ASD" --group "$group"
  python train_other_model.py --model_name NormFAAE --mode realistic --anomaly_ratio 0.02 --num_epochs 300 --data_name "ASD" --group "$group"
  python train_other_model.py --model_name FGANomaly --mode realistic --anomaly_ratio 0.02 --num_epochs 100 --data_name "ASD" --group "$group"
  for ratio in "${ratios[@]}"
  # simulated anomaly pollution case
  do
    python main.py --model_name PatchContrast --mode simulated --anomaly_ratio "$ratio" --num_epochs 101 --data_name "ASD" --group "$group" --remove_anomaly 1
    python train_other_model.py --model_name mtad_gat --mode simulated --anomaly_ratio "$ratio" --num_epochs 50 --data_name "ASD" --group "$group"
    python train_other_model.py --model_name MAUT --mode simulated --anomaly_ratio "$ratio" --num_epochs 100 --data_name "ASD" --group "$group"
    python train_other_model.py --model_name usad --mode simulated --anomaly_ratio "$ratio" --num_epochs 100 --data_name "ASD" --group "$group"
    python train_other_model.py --model_name cad --mode simulated --anomaly_ratio "$ratio" --num_epochs 30 --data_name "ASD" --group "$group"
    python train_other_model.py --model_name NormFAAE --mode simulated --anomaly_ratio "$ratio" --num_epochs 300 --data_name "ASD" --group "$group"
    python train_other_model.py --model_name FGANomaly --mode simulated --anomaly_ratio "$ratio" --num_epochs 100 --data_name "ASD" --group "$group"
  done
done


