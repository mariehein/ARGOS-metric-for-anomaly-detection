#!/usr/local_rwth/bin/zsh

### Job name
#SBATCH --job-name=best_hp_runs
#SBATCH --account rwth0934

#SBATCH --output=../output/hp_classifier/%J.txt

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
half_data=${6}
metric=${7}

echo "Signal number: $signal_number"
echo "Classifier: $classifier"
echo "Mode: $mode"
echo "Rotated: $rotated"
echo "Samples file: $samples_file"

#Needs to be changed for different cluster
gen_direc="/hpcwork/zu992399/model-agnostic-model-selection/${classifier}/optimized_hp/"
gen_hp_file="hp_files/optimized/${classifier}/"
hp_file_end="_hp_${metric}.yaml"

# build base directory depending on rotation
if [[ "$rotated" == "True" ]]; then
    direc="${gen_direc}${mode}_rotated"
    hp_file="${gen_hp_file}${mode}_rotated/Nsig_${signal_number}/"
    extra_args="--apply_random_rotation"
else
    direc="${gen_direc}${mode}"
    hp_file="${gen_hp_file}${mode}/Nsig_${signal_number}/"
    extra_args=""
fi

if [[ "$half_data" == "True" ]]; then
    direc="${direc}_half/${metric}/Nsig_${signal_number}/"
    extra_args="${extra_args} --use_half_statistics"
else
    direc="${direc}/${metric}/Nsig_${signal_number}/"
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
    --cl_filename ${hp_file_end} \
    --cl_file_direc ${hp_file} \
    ${extra_args}


