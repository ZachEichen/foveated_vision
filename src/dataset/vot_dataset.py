import logging
from os import path as Path

import torch
import time
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms.v2 import Resize

# /scratch/eecs542s001f23_class_root/eecs542s001f23_class/shared_data/foveated_vision/
# LaSOT dataset 
basedir_ = '/scratch/eecs542s001f23_class_root/eecs542s001f23_class/shared_data/group_raz/data/vot/'
directories = {'shortterm': Path.join(basedir_,'shortterm'),
               'longterm':Path.join(basedir_,'longterm')}

class VotDataset(Dataset):
    def __init__(self, dataset_name='longterm',targ_size = (650,650),clip_secs:int = 5):
        # get list of video sequences:
        self.basedir = directories[dataset_name]
        self.seq_len = int(clip_secs * 30)
        self.target_size = targ_size
        if targ_size is None:
            self.resize = lambda x:x
        else: 
            self.resize = Resize(size=targ_size,antialias=True)
        with open(Path.join(self.basedir,'sequences','list.txt'),'r') as seqfile: 
            videos = [line.strip() for line in seqfile.readlines() if len(line.strip()) > 0 ] 

        vidlengths = []
        for videoname in videos:
            viddir = Path.join(self.basedir,'sequences',videoname)
            with open(Path.join(viddir,'groundtruth.txt'),'r') as file: 
                vidlengths.append(sum(1 for line in file.readlines() if len(line) != 0))

        n_clips = [vl // self.seq_len for vl in vidlengths]
        self.videos =[ (v_name, i) for v_name, n_clips in zip(videos,n_clips) for i in range(n_clips)] 

        # self.cached_imgs = Path.join(self.basedir, 'img_caches')
        # self.cached_labels = Path.join(self.basedir, 'img_labels')
        # os.makedirs(self.cached_imgs,exist_ok=True)
        # os.makedirs(self.cached_labels,exist_ok=True)
        # for i in tqdm(range(len(self))): 
        #     self.preprocess_image(i,reprocess=False)
        
    def preprocess_image(self,imnum,reprocess:bool=False):
        if (reprocess or 
                (not Path.isfile(Path.join(self.cached_imgs,f'{imnum}.pt')) 
                or not Path.isfile(Path.join(self.cached_labels,f'{imnum}.pt')))):
            img, yval = self.get_vid(imnum)
            torch.save(img,Path.join(self.cached_imgs,f'{imnum}.pt'))
            torch.save(yval,Path.join(self.cached_labels,f'{imnum}.pt'))

    def __len__(self): 
        return len(self.videos)
    
    def get_vid(self, idx):
        vidname, pos = self.videos[idx]
        start_idx, end_idx = pos * self.seq_len, (pos+1) * self.seq_len 

        viddir = Path.join(self.basedir,'sequences',vidname)
        with open(Path.join(viddir,'groundtruth.txt'),'r') as file: 
            groundtruth = torch.tensor([[float(elt) for elt in line.split(',')] for line in file.readlines() if len(line) != 0])
        img = read_image(Path.join(viddir,f'color/{1:08d}.jpg'))
        first = self.resize(img)
        groundtruth = groundtruth[start_idx:end_idx,:] / torch.tensor([img.shape[-1],img.shape[-2],img.shape[-1],img.shape[-2]])
        # Groundtruth may have lines with [Nan, Nan, Nan, Nan], but we need labels at every timestep
        # TODO: interpolate the groundtruth labels
        # For now, just replicate the last non-Nan label
        # Last non-Nan label
        last_non_nan = torch.tensor([0,1.0,0,1.0])
        for i in range(groundtruth.shape[0]):
            if torch.any(torch.isnan(groundtruth[i])):
                # Replace the NaNs with the last non-NaN value
                groundtruth[i] = last_non_nan
            else:
                last_non_nan = groundtruth[i]

        images = torch.zeros([self.seq_len] + list(first.shape))
        for i in range(self.seq_len): 
            # Resize and remap colorspace to 0.0-1.0
            images[i] = self.resize(read_image(Path.join(viddir,f'color/{start_idx + i+1:08d}.jpg'))) / 255.0
        groundtruth = torch.stack([
            groundtruth[:,0], groundtruth[:,1], groundtruth[:,0] + groundtruth[:,2], groundtruth[:,1] + groundtruth[:,3]
        ],dim=-1)
        return images,groundtruth
    
    def __getitem__(self,idx):
        return self.get_vid(idx)



class NonNullVotDataset(Dataset):
    def __init__(self, dataset_name='longterm',targ_size = (650,650),clip_secs:int = 5):
        # get list of video sequences:
        self.basedir = directories[dataset_name]
        self.seq_len = int(clip_secs * 30)
        self.target_size = targ_size
        if targ_size is None:
            self.resize = lambda x:x
        else: 
            self.resize = Resize(size=targ_size,antialias=True)
        with open(Path.join(self.basedir,'sequences','list.txt'),'r') as seqfile: 
            videonames = [line.strip() for line in seqfile.readlines() if len(line.strip()) > 0 ] 

        self.videos = []
        for videoname in videonames:
            viddir = Path.join(self.basedir,'sequences',videoname)
            
            with open(Path.join(viddir,'groundtruth.txt'),'r') as file: 
                nonnulls = [('nan' not in line) for line in file.readlines() if len(line) != 0] 
            vid_segs = self._get_nonnull_segments(nonnulls,self.seq_len)
            self.videos.extend((videoname,startpoint) for startpoint in vid_segs)
            
    
    @staticmethod
    def _get_nonnull_segments(nonnulls,target_nframes):
        segment_start = None
        for i,val in enumerate(nonnulls): 
            if segment_start is None and val: 
                segment_start =i
            elif segment_start is not None and val and i-segment_start == target_nframes: 
                yield segment_start 
                segment_start = i 
            elif not val:
                segment_start =None
        if segment_start is not None and val and (i+1)-segment_start == target_nframes: 
            yield segment_start

    def __len__(self): 
        return len(self.videos)
    
    def get_vid(self, idx):
        vidname, start_frame = self.videos[idx]
        start_idx, end_idx = start_frame, start_frame + self.seq_len

        viddir = Path.join(self.basedir,'sequences',vidname)
        with open(Path.join(viddir,'groundtruth.txt'),'r') as file: 
            groundtruth = torch.tensor([[float(elt) for elt in line.split(',')] for line in file.readlines() if len(line) != 0])
        img = read_image(Path.join(viddir,f'color/{1:08d}.jpg'))
        first = self.resize(img)
        groundtruth = groundtruth[start_idx:end_idx,:] / torch.tensor([img.shape[-1],img.shape[-2],img.shape[-1],img.shape[-2]])
        # Groundtruth may have lines with [Nan, Nan, Nan, Nan], but we need labels at every timestep
        # TODO: interpolate the groundtruth labels
        # For now, just replicate the last non-Nan label
        # Last non-Nan label
        logging.debug(f'groundtruth label snippet {groundtruth}')
        logging.debug(f'groundtruth label snippet isna {torch.isnan(groundtruth)}')
        logging.debug(f'groundtruth label snippet isna {torch.any(torch.isnan(groundtruth),axis=1)}')
        assert(not torch.any(torch.isnan(groundtruth)))
        images = torch.zeros([self.seq_len] + list(first.shape))
        for i in range(self.seq_len): 
            # Resize and remap colorspace to 0.0-1.0
            images[i] = self.resize(read_image(Path.join(viddir,f'color/{start_idx + i+1:08d}.jpg'))) / 255.0
        groundtruth = torch.stack([
            groundtruth[:,0], groundtruth[:,1], groundtruth[:,0] + groundtruth[:,2], groundtruth[:,1] + groundtruth[:,3]
        ],dim=-1)
        return images,groundtruth
    
    def __getitem__(self,idx):
        return self.get_vid(idx)


def get_dataloader(dataset_name='longterm',targ_size = None,batch_size=3, clip_length_s=5, shuffle=True, seed = None,**loader_kwargs):
    collate_fn = None
    if seed is None: 
        seed = int(time.time()*1000)
        # print(seed)
    if 'generator' not in loader_kwargs: 
        ds = VotDataset(dataset_name=dataset_name,targ_size=targ_size, clip_secs=clip_length_s)
        gen = torch.Generator()
        loader_kwargs['generator'] = gen
    gen.manual_seed(seed)
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,**loader_kwargs,collate_fn=collate_fn)
    
if __name__ == "__main__": 
    # from tqdm import tqdm
    ds = NonNullVotDataset()
    logging.basicConfig(
        level=logging.INFO
    )  # Change this to INFO or WARNING to reduce verbosity, or DEBUG for max spam
    # print(ds)
    print(len(VotDataset()))
    print(len(ds))
    img, key = ds[4]
    for i in [1, 394, 123, 679,34,987,234,876]: 
        ds[i]
    print(img.shape)
    print(key.shape)
    # print(img, key)
    # print(f"Labels: {key}")

    dl = get_dataloader(targ_size=(650,650))
    print(dl.generator.initial_seed())
    time.sleep(1)
    dl2 = get_dataloader(targ_size=(650,650))
    print(dl2.generator.initial_seed())

    # for i,loaded in tqdm(enumerate(dl),total=len(ds)): 
        
    #     if i %100: 
    #         print(loaded)