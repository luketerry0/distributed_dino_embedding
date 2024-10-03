#!/usr/bin/env bash

#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --job-name=dino_embedding
#SBATCH --time=12:00:00
#SBATCH --output=/home/luketerry/distributed_dino_embedding/logs/%j_0_log.out
#SBATCH --error=/home/luketerry/distributed_dino_embedding/logs/%j_0_log.err
#SBATCH --mail-user=luke.h.terry-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=10
#SBATCH --partition=ai2es
#SBATCH --chdir=/home/luketerry/distributed_dino_embedding

EXPDIR=/home/luketerry/distributed_dino_embedding
cd /home/luketerry/distributed_dino_embedding

# using Dr. Fagg's conda setup script
. /home/fagg/tf_setup.sh
# activating a version of my environment
conda activate /home/jroth/.conda/envs/mct

srun python run_distributed_process.py \
    --embedding_path="/ourdisk/hpc/ai2es/luketerry/embeddings" \
    --data_path="/ourdisk/hpc/ai2es/jroth/data/NYSDOT_m4er5dez4ab/NYSDOT_m4er5dez4ab" \
    --num_gpus=4 \
    --size=4 \


