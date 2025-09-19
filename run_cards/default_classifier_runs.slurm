#!/usr/local_rwth/bin/zsh

### Job name
#SBATCH --job-name=default_runs
#SBATCH --account rwth0934

#SBATCH --output=../output/default/%J.txt

### Compute time --> calculated generously
#SBATCH --time=0-5:00:00
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
gen_direc="/hpcwork/zu992399/model-agnostic-model-selection/${classifier}/default_hp/"

# build base directory depending on rotation
if [[ "$rotated" == "True" ]]; then
    direc="${gen_direc}${mode}_rotated/Nsig_${signal_number}/"
    extra_args="--apply_random_rotation"
else
    direc="${gen_direc}${mode}/Nsig_${signal_number}/"
    extra_args=""
fi

# add samples_file if mode is cathode
if [[ "$mode" == "cathode" ]]; then
    extra_args="$extra_args --samples_file ${samples_file}"
fi

# final command
python run_pipeline.py \
    --classifier ${classifier} \
    --directory ${direc} \
    --signal_number ${signal_number} \
    --mode ${mode} \
    ${extra_args}


