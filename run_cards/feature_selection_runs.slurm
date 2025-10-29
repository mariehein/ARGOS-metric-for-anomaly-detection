#!/usr/local_rwth/bin/zsh

### Job name
#SBATCH --job-name=default_runs
#SBATCH --account rwth0934

#SBATCH --output=../output/select_features/%J.txt

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
half_data=${5}
features=${6}
samples_file=${7}

echo "Signal number: $signal_number"
echo "Classifier: $classifier"
echo "Mode: $mode"
echo "Rotated: $rotated"
echo "Half data: $half_data"
echo "features: $features"
echo "Samples file: $samples_file"

#Needs to be changed for different cluster
gen_direc="/hpcwork/zu992399/model-agnostic-model-selection/${classifier}/default_hp/"

# build base directory depending on rotation
if [[ "$rotated" == "True" ]]; then
    direc="${gen_direc}${mode}_rotated"
    extra_args="--apply_random_rotation"
else
    direc="${gen_direc}${mode}"
    extra_args=""
fi

if [[ "$half_data" == "True" ]]; then
    direc="${direc}_half"
    extra_args="${extra_args} --use_half_statistics"
fi

#if [[ "$useless" == "True" ]]; then
#    direc="${direc}_useless/Nsig_${signal_number}/"
#    extra_args="${extra_args} --unhelpful_features"
#else
#    direc="${direc}/Nsig_${signal_number}/"
#fi

if [[ "$features" != "baseline" ]]; then
    direc="${direc}_${features}/Nsig_${signal_number}/"
    if [[ "$features" == "DeltaR" ]]; then
        extra_args="${extra_args} --include_DeltaR"
    else
        extra_args="${extra_args} --input_set ${features}"
    fi
else
    direc="${direc}/Nsig_${signal_number}/"
fi

# add samples_file if mode is cathode
if [[ "$mode" == "cathode" ]]; then
    extra_args="$extra_args --samples_file ${samples_file}Nsig_${signal_number}/samples_inner.npy"
fi

# final command
python run_pipeline.py \
    --classifier ${classifier} \
    --directory ${direc} \
    --signal_number ${signal_number} \
    --mode ${mode} \
    ${extra_args}


