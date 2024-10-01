#!/usr/bin/env bash

groups=("real_satellite_data_1" "real_satellite_data_2")
ratios=("0.02" "0.05" "0.1" "0.15" "0.2")
ratio2=("1" "2" "3" "4")

for group in "${groups[@]}"
do
  for ratio in "${ratios[@]}"
  do
    # simulated pollution
    python main.py --model_name PatchContrast --mode simulated --anomaly_ratio "$ratio" --num_epochs 101 --data_name sate --group "$group" --remove_anomaly 1
    python train_other_model.py --model_name mtad_gat --mode simulated --anomaly_ratio "$ratio" --num_epochs 50 --data_name sate --group "$group"
    python train_other_model.py --model_name MAUT --mode simulated --anomaly_ratio "$ratio" --num_epochs 100 --data_name sate --group "$group"
    python train_other_model.py --model_name usad --mode simulated --anomaly_ratio "$ratio" --num_epochs 100 --data_name sate --group "$group"
    python train_other_model.py --model_name cad --mode simulated --anomaly_ratio "$ratio" --num_epochs 30 --data_name sate --group "$group"
    python train_other_model.py --model_name NormFAAE --mode simulated --anomaly_ratio "$ratio" --num_epochs 300 --data_name sate --group "$group"
    python train_other_model.py --model_name FGANomaly --mode simulated --anomaly_ratio "$ratio" --num_epochs 100 --data_name sate --group "$group"
    python train_other_model.py --model_name KMeans --mode simulated --anomaly_ratio "$ratio" --num_epochs 100 --data_name sate --group "$group"
  done

  for ratio in "${ratios2[@]}"
  do
  # realistic
    python main.py --model_name PatchContrast --mode realistic --window_length 1024 --anomaly_ratio "$ratio" --num_epochs 101 --data_name sate --group "$group" --remove_anomaly 1
    python train_other_model.py --model_name mtad_gat --mode realistic --anomaly_ratio "$ratio" --num_epochs 50 --data_name sate --group "$group"
    python train_other_model.py --model_name MAUT --mode realistic --anomaly_ratio "$ratio" --num_epochs 100 --data_name sate --group "$group"
    python train_other_model.py --model_name usad --mode realistic --anomaly_ratio "$ratio" --num_epochs 100 --data_name sate --group "$group"
    python train_other_model.py --model_name cad --mode realistic --anomaly_ratio "$ratio" --num_epochs 30 --data_name sate --group "$group"
    python train_other_model.py --model_name NormFAAE --mode realistic --anomaly_ratio "$ratio" --num_epochs 300 --data_name sate --group "$group"
    python train_other_model.py --model_name FGANomaly --mode realistic --anomaly_ratio "$ratio" --num_epochs 100 --data_name sate --group "$group"
    python train_other_model.py --model_name KMeans --mode realistic --anomaly_ratio "$ratio" --num_epochs 100 --data_name sate --group "$group"
  done
done


