# Model Agnostic Optimisation of Weakly Supervised Anomaly Detection

This repository contains the code for the following paper:

*How to pick the best anomaly detector?*
By Marie Hein, Gregor Kasieczka, Michael Krämer, Louis Moureaux, Alexander Mück and David Shih.
[arxiv:2511.14832](https://arxiv.org/abs/2511.14832)

## Structure of the repository
In addition to the code used to produce the paper runs, there are two folders in this repository: 
- In "hp_files", both default and optimized hyperparameters can be found, meaning that it is possible to reproduce the paper plots without running the full optimisation, which is the most computationally expensive part.
- In "run_cards", the run cards used to produce the paper runs are saved. Their structure is explained below. 

## Reproducing the paper results 
In order to make the reproduction of the paper plots easier, the run cards are available in "run_cards".

### Structure of the run cards
There are two run types: 
1. Default hyperparameter runs
2. Hyperparameter optimization runs

For the default hyperparameter runs, there is 
1. ```default_classifier_runs.slurm```, i.e. a slurm script, which starts ```run_pipeline.py``` with the arguments it was passed. Save directories are built automatically from the passed arguments. 
2. ```default_classifier_runs.sh```, i.e. a bash script, which loops over signal amounts, the different classifiers (NN, AdaBoost, HGB) and mode (IAD, cathode, cwola), as well as whether or not a random rotation should be applied to the data and passes them as arguments to the slurm scripts, which it submits. 

For the hyperparameter optimization, runs are more complex. There are  
1. multiple slurm scripts, namely: 
   1. ```hp_opt_runs.slurm```: Starting the hyperparameter optimization as an array job.
   2. ```hp_pick_best.slurm```: Picking the best hyperparameter runs after all array jobs are finished.
   3. ```hp_classifier_runs.slurm```: Producing classifier runs with half (see explanation in paper) and full statistics for the optimized hyperparameters for each metric after the job picking the best hyperparamter is complete.
2. ```hp_opt_runs.sh```, i.e. a bash script, which loops over all settings and submits slurm scripts.

In the version found in this github, all paper runs are submitted immediately. Depending on the computing resources available, it may be sensible/necessary to start runs a few at a time. This was done for the production runs for the paper.

### Adjusting to your system
To run on a different cluster/file system, several things need to be adjusted in the run cards: 
- Location of samples file for CATHODE 
- Run directory
- Adjust job submission file to submission system used on your computing cluster
- Activation of python environment

Additionally, the location of the LHCO files (specifically using version from [2309.13111](https://arxiv.org/abs/2309.13111), which can be found in [this Githb](https://github.com/uhh-pd-ml/treebased_anomaly_detection)) in your file system needs to be either set as default or passed as an argument to both ```run_pipeline.py```and ```run_hyperparameter_optimization.py``` as the parameters ```--data_file``` and ```--extrabkg_file```.

### Plotting the runs
The plotting notebook ```plotting.ipynb``` is kept as minimal as possible. If the save directories in the run cards are only adjusted by changing ```gen_direc``` in the slurm files, all plots should be reproducible by only adjusting ```general_directory``` in ```plotting.ipynb```. The detailed plotting functions can be found in ```plotting_utils_paper.py```. All necessary calculations as well as the construction of the regular plotting paths are performed in those functions based on the passed parameters. 


