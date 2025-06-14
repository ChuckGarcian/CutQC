#!/bin/bash
#SBATCH --output=reconstruction_example.out
#SBATCH --nodes=1                 # node count
#SBATCH --ntasks-per-node=2       # total number of tasks across all nodes
#SBATCH --cpus-per-task=12        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=40G                 # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:4
#SBATCH --time=00:01:55           # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin         # send email when job begins
#SBATCH --mail-type=end           # send email when job ends
#SBATCH --mail-type=fail          # send mail if job fails

# Setup for Multi-node Workload
export MASTER_PORT=$(get_free_port)  # Get a free Port
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)  # Master node is first host in list of hostnames
export MASTER_ADDR=$master_addr

# Sanity Print
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

# Load Modules
module purge
module load anaconda3/2024.2
conda activate CutQCSummer2025
module load gurobi/12.0.0

srun python reconstruction_example.py

