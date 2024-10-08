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


def run(dataloader, embedding_dir):
    """ embed this rank's data"""

    # get the model, according to this processes rank
    model = dino_model()
    for image, file in iter(dataloader):
        # get the dino embedding
        # im = image.to('cuda')
        embedding = dino_model()(image)
        # save the file somewhere
        directories = file[0].split('/')
        path = embedding_dir + '/' + directories[-2] + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(embedding, path + directories[-1][:-4] + '.pt') 

def dino_model():
    """get the dino model"""
    # os.environ['TORCH_HOME'] = '/scratch/jroth/'
    # os.environ['TORCH_HUB'] = '/scratch/jroth/'
    # DINOv2 vit-s (14) with registers
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', verbose=False)
    # state = model.state_dict()
    # mymodel = vit_small(14, 4)
    # mymodel.load_state_dict(state)
    model.eval()
    return model #.to(torch.device('cuda'))

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
    loader = DataLoader(dataset, shuffle=False, sampler=sampler)

    return loader

def init_process(fn, embedding_dir, data_path, backend='nccl'):
    """ Initialize the distributed environment. """

    dist.init_process_group(backend="nccl")
    dataloader = get_data_loader(data_path)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    fn(dataloader, embedding_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--embedding_dir", type=str, default='./embeddings')
    parser.add_argument("--data_dir", type=str, default='./20k_bronx/rdma/flash/hulk/raid/csutter/cron/data/NYSDOT_m4er5dez4ab/20230516')
    args = vars(parser.parse_args())
    init_process(run, args["embedding_dir"], args["data_dir"])