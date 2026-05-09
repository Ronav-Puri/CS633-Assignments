#!/bin/bash

# --- CONFIGURATION ---
P=8                       
M_VALUES=(262144 1048576) 
D1=2
D2=4
T=10
SEED=1000
EXECUTABLE="./src"

# Ensure executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Executable '$EXECUTABLE' not found. Compiling..."
    module load compiler/oneapi-2024/mpi
    mpicc src.c -o src -lm
fi

# Create output dir if missing
mkdir -p raw_results

# CLEANUP: Only remove tmp files for THIS specific P value so we don't delete other jobs
rm -f raw_results/res_P${P}_*.tmp

submit_job() {
    local CURRENT_M=$1
    local NODES=$(( (P + 15) / 16 )) # Calculate nodes needed

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=P${P}_M${CURRENT_M}
#SBATCH -N $NODES
#SBATCH --ntasks=$P
#SBATCH --ntasks-per-node=16
#SBATCH --output=out_P${P}_M${CURRENT_M}.txt
#SBATCH --partition=cpu
#SBATCH --time=00:10:00

module load compiler/oneapi-2024/mpi

# Temp file for this specific job
TMP_FILE="raw_results/res_P${P}_M${CURRENT_M}.tmp"
touch \$TMP_FILE

for i in {1..5}
do
   # Run and capture output
   output=\$(mpirun -np $P $EXECUTABLE $CURRENT_M $D1 $D2 $T $SEED  | tr ' ' ',')
   echo "$P,$CURRENT_M,\$output" >> \$TMP_FILE
done
EOT
}

# --- MAIN LOOP (Only over M) ---
for M in "${M_VALUES[@]}"; do

    # Concurrency Check: Wait if YOU (the user) have too many jobs in queue total
    while [ $(squeue -u $USER -h | wc -l) -ge 2 ]; do
        echo "Queue full (2+ jobs total). Waiting 15 seconds..."
        sleep 15
    done

    echo "Submitting job for P=$P, M=$M"
    submit_job $M

    sleep 2
done

echo "Submitted all jobs for P=$P."
