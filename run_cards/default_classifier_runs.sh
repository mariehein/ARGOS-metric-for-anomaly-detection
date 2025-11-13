signals=(0 50 100 150 200 250 300 400 500 600 700 800 900 1000)
classifier=("HGB" "NN" "AdaBoost")
mode=("IAD" "cwola" "cathode")
rotated=("False" "False" "False")

samples_direc=/hpcwork/zu992399/model-agnostic-model-selection/DE/baseline/CFM/
#need to replace samples_file for different cluster 

for ((index=0; index<${#mode}; index++)); do
    for s in ${signals}; do
        for c in ${classifier}; do
            sbatch default_classifier_runs.slurm ${s} ${c} ${mode[$((${index}+1))]} ${rotated[$((${index}+1))]} ${samples_direc}
        done
    done
done
