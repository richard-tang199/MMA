# MMA

This repository is the implementation of "MLP-Mixer based Masked Autoencoders Are Effective,Explainable and Robust for Time Series Anomaly Detection". We propose the MMA framework to achieve effective, explainable, and robust time series anomaly detection.

## Usage

### Setup

Installation

```shell
conda create -n MMA python=3.9
conda activate MMA
pip install -r requirements.txt
```

### Prepare datasets

Download datasets from this link: [ano_dataset](https://drive.google.com/drive/folders/1L2me0vvm39KsECm5P3FoFpRlGS0B75-t) and put them in the ano_dataset folder.

```
├─ano_dataset
├───ASD
├───synthetic
└───UCR
```

Visualization of the datasets is provided at https://drive.google.com/drive/folders/1ZmOJ-lAN0FfgDr6unwsU2LLubdQovv1x?usp=sharing

### Reproduce the effectiveness experiment results
Run our model

```shell
sh runners\run_asd.sh
sh runners\run_synthetic.sh
sh runners\run_ucr.sh
```

Run baseline models

```
sh runners\run_asd_other.sh
sh runners\run_synthetic_other.sh
sh runners\run_ucr_other.sh
```

For ablation experiments and parameter sensitivity analysis, please refer to the instructions in runners\run_asd.sh

### Reproduce the explainability experiment results

```shell
sh runners\run_asd_ex.sh
sh runners\run_synthetic_ex.sh
sh runners\run_ucr_ex.sh
```

### Reproduce the robustness experiment results

```shell
sh runners\run_asd_pollute.sh
sh runners\run_synthetic_pollute.sh
sh runners\run_ucr_pollute.sh
```

## Acknowledgement

We appreciate the following github repo very much for the valuable code base

https://github.com/PatchTST/PatchTST

https://github.com/ibm-granite/granite-tsfm
