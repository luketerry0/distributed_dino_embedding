torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=1 \
    distributed_process.py \
        --data_dir=./test_bronx \
        --embedding_dir=./embeddings