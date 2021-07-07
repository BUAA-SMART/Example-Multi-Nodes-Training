#!/bin/sh

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=2
#SBATCH -p dell
#SBATCH --gres=gpu:V100:2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=0
#SBATCH --time=0-02:00:00
#SBATCH -o ../logs/slurm.test


# -------------------------
# debugging flags (optional)
 export NCCL_DEBUG=INFO
 export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest cuda
# module load NCCL/2.4.7-1-cuda.10.0
# -------------------------

# run script from above
srun python3 ddp_trainer.py \
  --trainer.accelerator 'ddp' \
  --trainer.gpus 2 \
  --trainer.num_nodes 2 \
  --trainer.max_epochs 5
