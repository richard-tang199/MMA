# MMA
This repository is the implementation of "MLP-Mixer based Masked Autoencoders Are Effective,Explainable and Robust for Time Series Anomaly Detection". We propose the MMA framework to achieve effective, explainable, and robust time series anomaly detection. 
## MMA model architecture

<img src="imgs\figure4.png"/>

## Main results

###  Effectiveness

Results on multivariate datasets

<img src="imgs\mul_result.png"/>

Results on the univariate time series dataset: the UCR Archive

<img src="imgs\uni_result.png" style="zoom:50%;" />

###  Explainability

The explainability evaluation results on top 5 performing deep learning methods.

<img src="imgs\figure7.png" />

###  Robustness

The performance of models under different level of training set contamination

![](imgs\figure8.png)

### Case study

The detection results on the UCR dataset

![](imgs\figure9.png)
## Usage

### Setup

Installation

```shell
conda create -n MMA python=3.9
conda activate MMA
pip install -r requirements.txt
```

###  Prepare datasets

Download datasets from this link: [ano_dataset](https://drive.google.com/drive/folders/1vAujAC9cArVJRMBQbLJCSbZh-r77r6wi?usp=sharing) and put them in the ano_dataset folder.

```
├───ano_dataset
│   ├───ASD
│   ├───sate
│   ├───synthetic
└───└───UCR
```

Visualization of the datasets is provided at https://drive.google.com/drive/folders/1ZmOJ-lAN0FfgDr6unwsU2LLubdQovv1x?usp=sharing

### Reproduce the effectiveness experiment results

Run our model 

```shell
sh run_asd.sh
sh run_sate.sh
sh run_synthetic.sh
sh run_ucr.sh
```

Run baseline models

```
sh run_asd_other.sh
sh run_sate_other.sh
sh run_synthetic_other.sh
sh run_ucr_other.sh
```

### Reproduce the explainability experiment results

```shell
sh run_asd_ex.sh
sh run_sate_ex.sh
sh run_synthetic_ex.sh
sh run_ucr_ex.sh
```

### Reproduce the robustness experiment results

```shell
sh run_asd_pollute.sh
sh run_sate_pollute.sh
sh run_synthetic_pollute.sh
sh run_ucr_pollute.sh
```

## Acknowledgement 

We appreciate the following github repo very much for the valuable code base

https://github.com/PatchTST/PatchTST

https://github.com/ibm-granite/granite-tsfm


