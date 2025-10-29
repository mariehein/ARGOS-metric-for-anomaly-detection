signals=(0 50 100 150 200 250 300 400 500 600 700 800 900 1000)
# 900 1000)
features=("baseline")
rotated=("False")

for ((index=0; index<${#features}; index++)); do
    for s in ${signals}; do
        sbatch DE_runs.slurm ${s} ${features[$((${index}+1))]} ${rotated[$((${index}+1))]}
    done
done
