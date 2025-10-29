signals=(0 50 100 150 200 250 300 400 500 600 700 800 900 1000)
classifier=("NN")
mode=("IAD")
rotated=("False") # "False")
half=("True" "False")
features=("DeltaR" "extended1" "extended2" "extended3")
#features=("baseline")

samples_direc=/hpcwork/zu992399/model-agnostic-model-selection/DE/baseline/CFM/
#need to replace samples_file for different cluster 

for ((index=0; index<${#mode[@]}; index++)); do
    for s in ${signals}; do
        for c in ${classifier}; do
            for h in ${half}; do
                for f in ${features}; do
                    sbatch feature_selection_runs.slurm ${s} ${c} ${mode[$((${index}+1))]} ${rotated[$((${index}+1))]} ${h} ${f} ${samples_direc}
                done
            done
        done
    done
done
