#!/bin/bash

args=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12")

for g in "${args[@]}"
do
  python train_other_model.py --model_name mtad_gat --num_epochs 50 --data_name "ASD" --group "$g"
  python train_other_model.py --model_name mtgflow --num_epochs 40 --data_name "ASD" --group "$g"
  python train_other_model.py --model_name TranAD --num_epochs 50 --data_name "ASD" --group "$g"
  python train_other_model.py --model_name gdn --num_epochs 50 --data_name "ASD" --group "$g"
  python train_other_model.py --model_name FGANomaly --num_epochs 100 --data_name "ASD" --group "$g"
  python train_other_model.py --model_name NormFAAE --num_epochs 300 --data_name "ASD" --group "$g"
  python train_other_model.py --model_name MAUT --num_epochs 100 --data_name "ASD" --group "$g"
  python train_other_model.py --model_name "usad" --num_epochs 100 --data_name "ASD" --group "$g"
  python train_other_model.py --model_name "cad" --num_epochs 30 --data_name "ASD" --group "$g"
  python train_other_model.py --model_name "KMeans" --num_epochs 100 --data_name "ASD" --group "$g"
  python train_other_model.py --model_name "PatchAD" --num_epochs 10 --data_name "ASD" --group "$g"
done

