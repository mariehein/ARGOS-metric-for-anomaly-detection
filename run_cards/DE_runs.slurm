#!/usr/local_rwth/bin/zsh

### Job name
#SBATCH --job-name=DE_runs
#SBATCH --account rwth0934

#SBATCH --output=../output/DE/%J.txt

### Compute time --> calculated generously
#SBATCH --time=0-3:00:00
#SBATCH --mem=12G
#SBATCH --ntasks=1

#SBATCH --gres=gpu:1

### Activate environment
source /home/zu992399/.bashrc
conda activate pytorch-rocky9

## Go into run directory
cd ..
signal_number=${1}
features=${2}
rotated=${3}

echo "Signal number: $signal_number"
echo "Features: $features"
echo "Rotated: $rotated"

#Needs to be changed for different cluster
gen_direc="/hpcwork/zu992399/model-agnostic-model-selection/DE/"

# build base directory depending on rotation
if [[ "$rotated" == "True" ]]; then
    direc="${gen_direc}${features}_rotated/CFM/Nsig_${signal_number}/"
    extra_args="--apply_random_rotation"
else
    direc="${gen_direc}${features}/CFM/Nsig_${signal_number}/"
    extra_args=""
fi

num_blocks=4
frequencies=20
batch_size=4096
epochs=2000
hidden_dim=256

# final command
python run_DE.py \
    --input_set ${features} \
    --signal_number ${signal_number} \
    --directory ${direc} \
    --start_at_run 5  \
    --N_runs 10 \
    --DE_type "CFM" \
    --epochs ${epochs} --batch_size ${batch_size} \
    --frequencies ${frequencies} --hidden_dim ${hidden_dim} \
    --num_blocks ${num_blocks} \
    ${extra_args}


