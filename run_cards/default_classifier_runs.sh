signals=(0 50 100 150 200 250 300 400 500 600 700 800 900 1000)
classifier=("HGB" "AdaBoost" "NN")
mode=("IAD" "IAD" "cwola")
rotated=("False" "True" "False")

samples_file="/hpcwork/rwth0934/LHCO_dataset/ranit_samples/ensembling/window_scan_0/window_5/try_0/samples_SR.npy" 
#need to replace samples_file for different cluster 

for ((index=0; index<${#mode}; index++)); do
    for s in ${signals}; do
        for c in ${classifier}; do
            sbatch default_classifier_runs.slurm ${s} ${c} ${mode[$((${index}+1))]} ${rotated[$((${index}+1))]} ${samples_file}
        done
    done
done
