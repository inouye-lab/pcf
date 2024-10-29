# Counterfactual Fairness by Combining Factual and Counterfactual Predictions

## Introduction

Welcome! This is the official repository for the following paper:

>Zhou, Z., Liu, T., Bai, R., Gao, J., Kocaoglu, M., & Inouye, D. I.. Counterfactual Fairness by Combining Factual and Counterfactual Predictions. NeurIPS 2024.

## Dependency

Version of a few key packages are as below

numpy == 1.21.2
sklearn == 1.0.2
torch == 1.10.0

## Guidance

**Reproducing figures:** All figures shown in the paper can be reproduced by running the provided notebooks: `synthetic/demo_toy_regression.ipynb`,
`synthetic/demo_toy_classification.ipynb`, `/law/demo_cvae.estfinetune.ipynb`. 

We have provided the checkpoints of VAE required for generating figures. You can also retrain those models as instructed below.
In this project, we use [Weights And Biases](https://wandb.ai/) to log experiment and run sweeps. `entity` is removed for anonymization and you need to add your own wandb entity there.

### Semi-synthetic Preparation
The first step is to generate ground truth CGM. 

To recreate the experiments, please first traverse to the directory 
```$ cd VAE```, then run the following command to create the Weights And Biases sweep (If you are not logged into `wandb` you will have to log in and re-run this command):

```$ wandb sweep configs/law_gt.yaml```


This should print a command which you can use to run a wandb agent (e.g., `wandb agent "xxx/NeurIPS24-CVAE/yyy"`).
To begin running experiments you can just enter the wandb agent command and this should begin automatically creating and tracking all the experiments.

The next step is to get estimated VAE. Run the following sweep under the directory `VAE` 
```$ wandb sweep configs/law.yaml```.

## Notes 
The result with DCEVAE on Adult can be found in the branch `dcevae`. 
We created a separate branch due to some difference in model and experiment setup. 

## Citation
If you find this code useful, we would be grateful if you cite our [paper](https://openreview.net/forum?id=J0Itri0UiN)
```
@inproceedings{zhou2024pcf,
  author       = {Zeyu Zhou and
                  Tianci Liu and
                  Ruqi Bai and 
                  Jing Gao and 
                  Murat Kocaoglu and 
                  David I. Inouye},
  title        = {Counterfactual Fairness by Combining Factual and
Counterfactual Predictions},
  journal      ={Advances in Neural Information Processing Systems},
  year         = {2024},
}
```