#!/usr/local_rwth/bin/zsh

### Job name
#SBATCH --job-name=hp_get_best
#SBATCH --account rwth0934

#SBATCH --output=../output/hp_pick_best/%J.txt

### Compute time --> calculated generously
#SBATCH --time=0-0:30:00
#SBATCH --mem=12G
#SBATCH --ntasks=1

### Activate environment
source /home/zu992399/.bashrc
conda activate cathode

## Go into run directory
cd ..
signal_number=${1}
classifier=${2}
mode=${3}
rotated=${4}
samples_file=${5}

echo "Signal number: $signal_number"
echo "Classifier: $classifier"
echo "Mode: $mode"
echo "Rotated: $rotated"
echo "Samples file: $samples_file"

#Needs to be changed for different cluster
gen_direc="/hpcwork/zu992399/model-agnostic-model-selection/${classifier}/run_optimization/"
gen_hp_direc="hp_files/optimized/${classifier}/"

# build base directory depending on rotation
if [[ "$rotated" == "True" ]]; then
    direc="${gen_direc}${mode}_rotated/Nsig_${signal_number}/"
    hp_direc="${gen_hp_direc}${mode}_rotated/Nsig_${signal_number}/"
else
    direc="${gen_direc}${mode}/Nsig_${signal_number}/"
    hp_direc="${gen_hp_direc}${mode}/Nsig_${signal_number}/"
fi

# final command
python pick_best_hyperparameters.py \
    --directory ${direc} \
    --file_save_directory ${hp_direc}
    ${extra_args}