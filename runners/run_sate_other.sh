#!/usr/bin/env bash

groups=("real_satellite_data_1" "real_satellite_data_2" "real_satellite_data_3")

for g in "${groups[@]}"
do
  python train_other_model.py --model_name mtad_gat --num_epochs 50 --data_name sate --group "$g"
  python train_other_model.py --model_name mtgflow --num_epochs 40 --data_name sate --group "$g"
  python train_other_model.py --model_name TranAD --num_epochs 50 --data_name sate --group "$g"
  python train_other_model.py --model_name gdn --num_epochs 50 --data_name sate --group "$g"
  python train_other_model.py --model_name FGANomaly --num_epochs 100 --data_name sate --group "$g"
  python train_other_model.py --model_name NormFAAE --num_epochs 300 --data_name sate --group "$g"
  python train_other_model.py --model_name MAUT --num_epochs 100 --data_name sate --group "$g"
  python train_other_model.py --model_name "usad" --num_epochs 100 --data_name sate --group "$g"
  python train_other_model.py --model_name cad --num_epochs 30 --data_name sate --group "$g"
done

