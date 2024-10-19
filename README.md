# MMA

This repository is the implementation of "MLP-Mixer based Masked Autoencoders Are Effective,Explainable and Robust for Time Series Anomaly Detection". We propose the MMA framework to achieve effective, explainable, and robust time series anomaly detection.

## MMA model architecture

<div align=center>
<img src="imgs\figure4.png" width="90%">
</div>

## Main results

### Effectiveness

Results on multivariate datasets

<div align=center>
<img src="imgs\mul_result.png" />
</div>

Results on the univariate time series dataset: the KDD21 dataset

<div align=center>
<img src="imgs\uni_result_new.png" width="50%" />
</div>

### Explainability

The explainability evaluation results on top 5 performing deep learning methods.

<div align=center>
<img src="imgs\explain.png">
</div>

### Robustness

The performance of models under different level of training set contamination

<div align=center>
<img src="imgs\robustness.png">
</div>

### Case study

The detection results on the UCR dataset

<div align=center>
<img src="imgs\cases.png">
</div>

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
├───sate   
├───synthetic
└───UCR
```

Visualization of the datasets is provided at https://drive.google.com/drive/folders/1ZmOJ-lAN0FfgDr6unwsU2LLubdQovv1x?usp=sharing

### Reproduce the effectiveness experiment results

It is worth noting that when writing the code, we defined the model name as PatchContrast. However, when writing the paper, we found that this name was too long, so we renamed it to MMA. The table below shows the corresponding model names in the paper and in the code for ablation studies.

| Name in paper   | Name in code   |
| :-------------- | :------------- |
| MMA             | PatchContrast  |
| MMA_GRU         | PatchGru       |
| MMA_Transformer | PatchAttention |
| MMA w/o CL      | PatchDetector  |

Run our model

```shell
sh runners\run_asd.sh
sh runners\run_sate.sh
sh runners\run_synthetic.sh
sh runners\run_ucr.sh
```

Run baseline models

```
sh runners\run_asd_other.sh
sh runners\run_sate_other.sh
sh runners\run_synthetic_other.sh
sh runners\run_ucr_other.sh
```

For ablation experiments and parameter sensitivity analysis, please refer to the instructions in runners\run_asd.sh

### Reproduce the explainability experiment results

```shell
sh runners\run_asd_ex.sh
sh runners\run_sate_ex.sh
sh runners\run_synthetic_ex.sh
sh runners\run_ucr_ex.sh
```

### Reproduce the robustness experiment results

```shell
sh runners\run_asd_pollute.sh
sh runners\run_sate_pollute.sh
sh runners\run_synthetic_pollute.sh
sh runners\run_ucr_pollute.sh
```

## Acknowledgement

We appreciate the following github repo very much for the valuable code base

https://github.com/PatchTST/PatchTST

https://github.com/ibm-granite/granite-tsfm
