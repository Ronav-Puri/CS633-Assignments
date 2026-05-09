# CS633-Assignments
Assignments which were part of the course CS633: Parallel Computing, during my sixth semester at IITK.

**Group Members**: Aarsh Jain, Bikramjeet Singh, Hitarth Makawana, Ronav Puri, Saksham Verma

---

## Repository Structure

```text
.
├── Assignment 1/
│   ├── slurm_outputs/             # Raw output logs from the cluster
│   ├── CS633_ass1.pdf             # Original problem statement
│   ├── CS633_ass1_report.pdf      # Detailed performance and scaling analysis report
│   ├── job8.sh                    # Execution script for 8 processes
│   ├── job16.sh                   # Execution script for 16 processes
│   ├── job32.sh                   # Execution script for 32 processes
│   ├── plot.py                    # Python script to generate boxplots from timing data
│   ├── redirect_tmp_to_csv.sh     # Bash utility to aggregate .tmp logs into a CSV
│   ├── src.c                      # MPI C source code for 1D Data Exchange
│   └── timings.csv                # Aggregated execution times
│
└── Assignment 2/
    ├── slurm_outputs/             # Raw output logs from the cluster
    ├── CS633_ass2.pdf             # Original problem statement
    ├── CS633_ass2_report.pdf      # Detailed performance and scaling analysis report
    ├── final_timing.csv           # Aggregated execution times
    ├── job.sh                     # Execution script for cluster deployment
    ├── plot.py                    # Python script to generate boxplots from timing data
    └── src.c                      # MPI C source code for 3D Stencil Computation
```
---

### Assignment 1: 1D Point-to-Point Data Exchange
This assignment explores parallel point-to-point communication protocols and their performance bottlenecks. The program simulates a data exchange where a sender rank transmits a large array of ```doubles``` to two specific receiver ranks located at fixed logical distances (```D1``` and ```D2```). The receivers perform localized computations (squaring and logarithmic operations) and return the data.

**Key Features:**
* Point-to-point MPI communication (```MPI_Send```, ```MPI_Recv```).
* Avoidance of communication serialization by alternating send/receive schedules based on process groups.
* Weak scaling performance analysis across 8, 16, and 32 processes, highlighting the impact of inter-node vs. intra-node communication overhead.

### Assignment 2: 3D Stencil Computation & Isovalue Detection
This assignment tackles a much more complex 3D domain decomposition problem. The global grid is divided into a Cartesian topology where each process manages a local sub-domain. The program performs a ```d```-point stencil computation (averaging neighboring cells) and simultaneously detects "isovalues" (crossings across a threshold) without double-counting inter-process boundaries.

**Key Features:**
* Advanced memory management flattening a 4D array (Z, Y, X, Fields) into a 1D structure.
* Highly optimized Non-Blocking Halo Exchanges using ```MPI_Isend``` and ```MPI_Irecv```.
* Utilization of MPI Derived Datatypes (```MPI_Type_indexed```) to send entire non-contiguous 3D boundary faces in a single logical message.
* Bulk synchronous reduction to minimize global synchronization bottlenecks.

---

## How to Run

### Prerequisites
* An MPI Implementation (e.g., OpenMPI, MPICH)
* A C compiler (```gcc``` or ```mpicc```)
* Python 3.x with ```matplotlib``` and ```pandas``` (for generating plots)
* A SLURM-based cluster environment (optional, but rquired to use the provided ```.sh``` job scripts)

### Running Assignment 1
1. Navigate to the directory: ```cd "Assignment 1"```
2. Compile the code: ```mpicc src.c -o src -lm```
3. Execute the job scripts (if on a SLURM cluster):
   ```
   sbatch job8.sh
   sbstch job16.sh
   sbatch job32.sh
   ```
   (Alternatively, run manually: ```mpirun -np <P> ./src <M> <D1> <D2> <T> <seed>```)
4. Aggregate results and plot:
   ```
   ./redirect_tmp_to_csv.sh
   python plot.py
   ```

### Running Assignment 2
1. Navigate to the directory: ```cd "Assignment 2"```
2. Compile the code: ```mpicc src.c -o src -lm```
3. Execute the job scripts (if on a SLURM cluster): ```sbatch job.sh```\
   (Alternatively, run manually: ```mpirun -np <P> ./src <d> <ppn> <px> <py> <pz> <nx> <ny> <nz> <T> <seed> <F> <isovalue>```)
5. Generate plots: ```python plot.py```
