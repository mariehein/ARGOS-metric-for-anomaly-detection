#signals=(0 50 100 150 200 250 300 400 500 600 700 800 900 1000)

signals=(150)
#signals=(0 50 100 150 200 250 300 400 500 600 700 800 900 1000) # AdaBoost IAD rotated running
#signals=(0 50 100 150 200 250 300 400 500 600 700 800 900 1000) # NN IAD rotated running
#signals=(0 50 100 150 200 250 300 400 500 600 700 800 900 1000) # AdaBoost cwola running


classifier=("AdaBoost")

#mode=("cwola")
mode=("IAD")
# "IAD" "cathode" "cwola")
rotated=("True")
# "True" "False" "False")
metrics=("val_SIC" "max_SIC" "val_loss")

samples_file="/hpcwork/rwth0934/LHCO_dataset/ranit_samples/ensembling/window_scan_0/window_5/try_0/samples_SR.npy" 
#need to replace samples_file for different cluster 

for ((index=0; index<${#mode}; index++)); do
    for s in ${signals}; do
        for c in ${classifier}; do
            #jid1=$(sbatch --parsable hp_opt_runs.slurm ${s} ${c} ${mode[$((${index}+1))]} ${rotated[$((${index}+1))]} ${samples_file})
            #jid2=$(sbatch --dependency=afterany:${jid1} --parsable hp_pick_best.slurm ${s} ${c} ${mode[$((${index}+1))]} ${rotated[$((${index}+1))]} ${samples_file})
            #jid2=$(sbatch --parsable hp_pick_best.slurm ${s} ${c} ${mode[$((${index}+1))]} ${rotated[$((${index}+1))]} ${samples_file})
            jid2=$(sbatch --dependency=afterany:61533064 --parsable hp_pick_best.slurm ${s} ${c} ${mode[$((${index}+1))]} ${rotated[$((${index}+1))]} ${samples_file})
            for m in ${metrics}; do
                #sbatch hp_classifier_runs.slurm ${s} ${c} ${mode[$((${index}+1))]} ${rotated[$((${index}+1))]} ${samples_file} "True" ${m}
                #sbatch hp_classifier_runs.slurm ${s} ${c} ${mode[$((${index}+1))]} ${rotated[$((${index}+1))]} ${samples_file} "False" ${m}
                sbatch --dependency=afterany:${jid2} hp_classifier_runs.slurm ${s} ${c} ${mode[$((${index}+1))]} ${rotated[$((${index}+1))]} ${samples_file} "True" ${m}
                sbatch --dependency=afterany:${jid2} hp_classifier_runs.slurm ${s} ${c} ${mode[$((${index}+1))]} ${rotated[$((${index}+1))]} ${samples_file} "False" ${m}
            done
        done
    done
done