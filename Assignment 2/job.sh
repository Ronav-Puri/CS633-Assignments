#!/bin/bash

# --- CONFIGURATION ---
NX_NY_NZ_VALUES=(120 240)
P_VALUES=(32 48 64 96)
D=7
T=5
SEED=1000
F=2
ISOVALUE=500
EXECUTABLE="./src"

rm -f src
# 1. Compile with High Optimization
module load compiler/oneapi-2024/mpi
mpicc src.c -o src

if [ ! -f $EXECUTABLE ]; then
    echo "Error: Compilation failed!"
    exit 1
fi

rm -rf slurm_outputs raw_results final_results.csv
# 2. Setup Directories
mkdir -p slurm_outputs
mkdir -p raw_results

# Function to submit a job for a specific P and N configuration
submit_job() {
    local P=$1
    local N=$2
    
    # Assignment Specific Grid Logic
    if [ $P -eq 32 ]; then px=4; py=4; pz=2; ppn=32; nodes=1; fi
    if [ $P -eq 48 ]; then px=6; py=4; pz=2; ppn=48; nodes=1; fi
    if [ $P -eq 64 ]; then px=4; py=4; pz=4; ppn=32; nodes=2; fi
    if [ $P -eq 96 ]; then px=6; py=4; pz=4; ppn=48; nodes=2; fi

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=P${P}_N${N}
#SBATCH -N $nodes
#SBATCH --ntasks=$P
#SBATCH --ntasks-per-node=$ppn
#SBATCH --output=slurm_outputs/out_P${P}_N${N}.txt
#SBATCH --partition=cpu
#SBATCH --time=00:15:00

module load compiler/oneapi-2024/mpi

for i in {1..5}
do
   full_output=\$(mpirun -np $P $EXECUTABLE $D $ppn $px $py $pz $N $N $N $T $SEED $F $ISOVALUE)

   echo "=== Output for P=$P, N=$N, Run=\$i ==="
   echo "\$full_output"
   echo "========================================"

   exec_time=\$(echo "\$full_output" | tail -n 1)

   echo "$P,$N,\$i,\$exec_time" >> raw_results/res_P${P}_N${N}.tmp
done
EOT
}

# --- MAIN SUBMISSION LOOP ---
for P in "${P_VALUES[@]}"; do
    for N in "${NX_NY_NZ_VALUES[@]}"; do
        
        # Concurrency Control (Strictly max 2 jobs in queue)
        while [ $(squeue -u $USER -h | wc -l) -ge 2 ]; do
            echo "Queue full (2 jobs active). Waiting 15 seconds..."
            sleep 15
        done

        echo "Submitting: P=$P, N=$N"
        submit_job $P $N
        
        # Give SLURM 5 seconds to fully register the job in the queue
        sleep 5 
    done
done

echo "All jobs submitted! Waiting for final jobs to finish..."
while [ $(squeue -u $USER -h | wc -l) -gt 0 ]; do
    sleep 15
done

# --- AGGREGATION ---
#echo "P,N,Run,Time" > final_timing.csv
#cat raw_results/*.tmp >> final_timing.csv
#echo "Done! Data for boxplots saved in final_timing.csv"

echo "P,N,Run,Time" > final_timing.csv

awk -F',' 'NF==4 && $4 ~ /^[0-9.]+$/' raw_results/*.tmp >> final_timing.csv

echo "Done! Clean data saved in final_timing.csv"
