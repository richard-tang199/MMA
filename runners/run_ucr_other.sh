#!/usr/bin/env bash

for g in {1..250}
do
  python train_other_model.py --model_name DAMP --num_epochs 50 --data_name "UCR" --group "$g"
  python train_other_model.py --model_name MP --num_epochs 50 --data_name "UCR" --group "$g"
  python train_other_model.py --model_name TranAD --num_epochs 50 --data_name "UCR" --group "$g"
  python train_other_model.py --model_name MAUT --mode normal --num_epochs 100 --data_name "UCR" --group "$g"
  python train_other_model.py --model_name cad --mode normal --num_epochs 30 --data_name "UCR" --group "$g"
  python train_other_model.py --model_name mtad_gat --num_epochs 50 --data_name "UCR" --group "$g"
  python train_other_model.py --model_name FGANomaly --num_epochs 100 --data_name "UCR" --group "$g"
  python train_other_model.py --model_name usad --num_epochs 100 --data_name "UCR" --group "$g"
  python train_other_model.py --model_name KMeans --num_epochs 100 --data_name "UCR" --group "$g"
  python train_other_model.py --model_name PatchAD --num_epochs 10 --data_name "UCR" --group "$g"
  python train_tsb.py --model_name SAND --num_epochs 50 --data_name "UCR" --group "$g"
  python train_tsb.py --model_name Series2Graph --num_epochs 50 --data_name "UCR" --group "$g"
done