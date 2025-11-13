signals=(0 50 100 150 200 250 300 400500 600 700 800 900 1000)
classifier=("HGB" "NN" "AdaBoost")
mode=("IAD" "cwola" "cathode")
rotated=("False" "False" "False")
metrics=("val_SIC" "max_SIC" "val_loss")

samples_direc=/hpcwork/zu992399/model-agnostic-model-selection/DE/baseline/CFM/
#need to replace samples_file for different cluster 

for ((index=0; index<${#mode}; index++)); do
    for s in ${signals}; do
        for c in ${classifier}; do
            jid1=$(sbatch --parsable hp_opt_runs.slurm ${s} ${c} ${mode[$((${index}+1))]} ${rotated[$((${index}+1))]} ${samples_direc})
            jid2=$(sbatch --dependency=afterany:${jid1} --parsable hp_pick_best.slurm ${s} ${c} ${mode[$((${index}+1))]} ${rotated[$((${index}+1))]} ${samples_direc})
            #jid2=$(sbatch --parsable hp_pick_best.slurm ${s} ${c} ${mode[$((${index}+1))]} ${rotated[$((${index}+1))]} ${samples_file})
            for m in ${metrics}; do
                #sbatch hp_classifier_runs.slurm ${s} ${c} ${mode[$((${index}+1))]} ${rotated[$((${index}+1))]} ${samples_file} "True" ${m}
                #sbatch hp_classifier_runs.slurm ${s} ${c} ${mode[$((${index}+1))]} ${rotated[$((${index}+1))]} ${samples_file} "False" ${m}
                sbatch --dependency=afterany:${jid2} hp_classifier_runs.slurm ${s} ${c} ${mode[$((${index}+1))]} ${rotated[$((${index}+1))]} ${samples_direc} "True" ${m}
                sbatch --dependency=afterany:${jid2} hp_classifier_runs.slurm ${s} ${c} ${mode[$((${index}+1))]} ${rotated[$((${index}+1))]} ${samples_direc} "False" ${m}
            done
        done
    done
done