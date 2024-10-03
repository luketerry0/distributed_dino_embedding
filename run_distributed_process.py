from argparse import ArgumentParser
from pathlib import Path
from distributed_process import init_process, run
import torch.multiprocessing as mp



ROOT = Path().resolve()
MEMORY_LIMIT = 2e8

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--embedding_path", type=str, required=True)
    parser.add_argument("--size", type=int, required=True)
    
    args = vars(parser.parse_args())

    processes = []
    mp.set_start_method("spawn")
    size = args["size"]
    
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run, args["embedding_path"]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()