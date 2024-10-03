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



def run(rank, size, dataloader, model, embedding_dir):
    """ embed this rank's data"""
    for image, file in iter(dataloader):
        # get the dino embedding
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
    return model.to('cpu')

def get_data_loader():
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
    dataset = StreetImageDataset("./20k_bronx/rdma/flash/hulk/raid/csutter/cron/data/NYSDOT_m4er5dez4ab/20230516", transform = transform)
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, shuffle=False, sampler=sampler)

    return loader

def init_process(rank, size, fn, embedding_dir, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend, rank=rank, world_size=size)
    dataloader = get_data_loader()
    model = dino_model()

    
    fn(rank, size, dataloader, dino_model, embedding_dir)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    embedding_dir = "./embeddings"

    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run, embedding_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()