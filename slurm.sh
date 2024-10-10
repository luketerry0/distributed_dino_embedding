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

# get the IP of the node used for the master process
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | grep -oE "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b")


EXPDIR=/home/luketerry/distributed_dino_embedding
cd /home/luketerry/distributed_dino_embedding

# using Dr. Fagg's conda setup script
. /home/fagg/tf_setup.sh
# activating a version of my environment
conda activate /home/jroth/.conda/envs/mct

torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc-per-node=4 \
    distributed_process.py \
        --data_dir=/ourdisk/hpc/ai2es/jroth/data/NYSDOT_m4er5dez4ab/NYSDOT_m4er5dez4ab \
        --embedding_dir=/ourdisk/hpc/ai2es/luketerry/batched_embeddings

    # --rdzv_id $RANDOM \
    # --rdzv_backend c10d \
    # --rdzv_endpoint "$head_node_ip:64425" \
    # --standalone \

