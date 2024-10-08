import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from street_image_dataset import StreetImageDataset
from torchvision.transforms import v2
import torchvision.transforms as transforms
import torchvision
import pickle
from argparse import ArgumentParser
from pathlib import Path
import uuid

@torch.no_grad()
def run(dataloader, embedding_dir, model):
    """ embed this rank's data"""
    # get the model, according to this processes rank
    for image, file in iter(dataloader):
        # get the dino embedding
        init_memory = torch.cuda.memory_allocated()
        im = image.to('cuda')
        init_memory = torch.cuda.memory_allocated()
        embedding = model(im)
        del im

        # save the embeddings and image filenames somewhere
        stamp = str(uuid.uuid1()) # uuid generated based on time and node (should be resistant to collision)
        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir)
        torch.save(embedding, embedding_dir + '/' + stamp + '.pt') 
        del embedding

        with open(embedding_dir + '/' + stamp + '_filenames.pickle', 'wb') as f:
            print(len(file))
            pickle.dump(file, f)
        torch.cuda.empty_cache()

@torch.no_grad()
def dino_model():
    """get the dino model"""
    # DINOv2 vit-s (14) with registers
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', verbose=False)
    model.eval()
    for param in model.parameters():
        param.grad = None
    model = model.to(torch.device('cuda'))
    # print(f"model size: {torch.cuda.memory_allocated()}")
    return model

def get_data_loader(path):
    """ get the data loader"""
    transform = v2.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        transforms.Resize(size=(256, 256), antialias=True),
                        transforms.CenterCrop((224, 224)),
                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]
                                            ),
                    ]
                    )
    dataset = StreetImageDataset(path, transform = transform)
    sampler = DistributedSampler(dataset)


    loader = DataLoader(dataset, shuffle=False, sampler=sampler, batch_size=64)

    return loader

def init_process(fn, embedding_dir, data_path, backend='nccl'):
    """ Initialize the distributed environment. """
    model = dino_model()
    dist.init_process_group(backend="nccl")
    dataloader = get_data_loader(data_path)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    fn(dataloader, embedding_dir, model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--embedding_dir", type=str, default='./embeddings')
    parser.add_argument("--data_dir", type=str, default='./20k_bronx/rdma/flash/hulk/raid/csutter/cron/data/NYSDOT_m4er5dez4ab/20230516')
    
    args = vars(parser.parse_args())
    init_process(run, args["embedding_dir"], args["data_dir"])
    dist.destroy_process_group()