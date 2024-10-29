# Counterfactual Fairness by Combining Factual and Counterfactual Predictions

## Note 
This is the branch where we trained the DCEVAE on Adult following the origial setup in [Zuo23](https://github.com/osu-srml/CF_Representation_Learning/). 
A lot of code is copied from their repo (mainly the training and model script) for the sake of the correctness of reproducing their results.

Note that the setup is not exactly the same as that with CVAE on Law in the main paper.

## Guidance

**Reproducing figures:** All figures shown in the paper can be reproduced by running the provided notebooks: `/adult/demo_dcevae.ipynb`. 

We have provided the checkpoints of VAE required for generating figures. You can also retrain those models as instructed below.
In this project, we use [Weights And Biases](https://wandb.ai/) to log experiment and run sweeps. `entity` is removed for anonymization and you need to add your own wandb entity there.

### Semi-synthetic Preparation
The first step is to generate ground truth CGM. 

To recreate the experiments, please first traverse to the directory 
```$ cd VAE```, then run the following command to create the Weights And Biases sweep (If you are not logged into `wandb` you will have to log in and re-run this command):

```$ wandb sweep configs/adult_gt.yaml```


This should print a command which you can use to run a wandb agent (e.g., `wandb agent "xxx/NeurIPS24-CVAE/yyy"`).
To begin running experiments you can just enter the wandb agent command and this should begin automatically creating and tracking all the experiments.

The next step is to get estimated VAE. Run the following sweep under the directory `VAE` 
```$ wandb sweep configs/adult.yaml```.

