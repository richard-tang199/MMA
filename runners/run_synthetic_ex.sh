#!/bin/bash

python explain.py --model_name PatchContrast --data_name "synthetic"
python explain.py --model_name mtad_gat --data_name "synthetic"
python explain.py --model_name TranAD --data_name "synthetic"
python explain.py --model_name MAUT --data_name "synthetic"
python explain.py --model_name cad --data_name "synthetic"


