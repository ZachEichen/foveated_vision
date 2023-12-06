import torch 
import sys
from utils import * 
import subprocess
from os.path import join
import os 
import logging 
torchvision.disable_beta_transforms_warning()

from dataset.vot_dataset import *
from peripheral_foveal_vision_model import PeripheralFovealVisionModel


def visualize_video(model, video_frames,gt_bounding_box,targ_dir=None,draw_bbox=True):
    model.eval()
    with torch.no_grad(): 
        bboxes, fixations = model(video_frames)
    
    to_dir = 'out'
    if targ_dir is not None: 
        to_dir = join('out',targ_dir)
        os.makedirs(to_dir,exist_ok=True)

    if draw_bbox: 
        names = ['predicted-bb','fovea','ground truth bb']
        bboxes = [bboxes,fixations,gt_bounding_box]
        normby = ['default','default','xyxy']
    else: 
        names = ['fovea','ground truth bb']
        bboxes = [fixations,gt_bounding_box]
        normby = ['default','xyxy']

    imgs_with_bboxes = draw_bboxes(video_frames,bboxes,names=names,norm_by=normby)
    for i in range(video_frames.shape[0]): 
        torchvision.io.write_jpeg(imgs_with_bboxes[i,...],join(to_dir,f'bbox_{i}.jpg'))
    # this assumes you have ffmpeg on your system and added to your path 
    # to install it you can run the following commands: (in an appropriate directory )
    # this downloads a precompiled binary (linked to from the official ffmpeg project page) unpacks it and adds it to your path
    #       wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
    #       tar xJfv ffmpeg-release-amd64-static.tar.xz
    #       rm ffmpeg-release-amd64-static.tar.xz
    #       chmod +x ffmpeg-6.1-amd64-static/ffmpeg
    #       echo "export PATH=$(pwd)/ffmpeg-6.1-amd64-static:\$PATH" >>~/.bashrc
    #       source ~/.bashrc
    #       set FFMPEG_PATH in ENV.py to your actual path value 
    try: 
        from ENV import FFMPEG_PATH

        subprocess.run([FFMPEG_PATH,'-y','-f','image2','-i',join(to_dir,'bbox_%d.jpg'), join(to_dir,'bboxes.mp4')])
    except ImportError:
        print('skipping video production')


def main(model_path,targdir='sample_{}',choose_vids=[9,567,293,104,871,32,712,255, 2 ,15, 74, 204,314,370],device='cpu'):

    model = PeripheralFovealVisionModel()
    model.load_state_dict(torch.load(model_path,map_location=torch.device(device)))
    # _, test_loader = get_train_test_dataloaders(seed=1,batch_size=3,targ_size=(224, 224),clip_length_s=5)
    ds = NonNullVotDataset()
    # ds = test_loader.dataset
    print(len(ds))
    # print(ds[choose_vid][1])
    for choose_id in choose_vids: 
        targdir_fmt = targdir.format(choose_id)
        vid, label = ds[choose_id]
        visualize_video(model,vid, label,targ_dir=targdir_fmt,draw_bbox=True)

if __name__ == "__main__":
    basedir = '/scratch/eecs542s001f23_class_root/eecs542s001f23_class/shared_data/group_raz/models/'
    name = '20231206_072942_model_epoch_2_step_2080.pth' 

    # basedir = 'models'
    # name = '20231206_051901_model_epoch_1_step_1360_loss_0.0013735051034018397.pth'
    if len(sys.argv) ==3: 
        basedir = sys.argv[1]
        name = sys.argv[2]
    elif len(sys.argv) == 2: 
        name = sys.argv[1]

    print(f'\n\n\n USING MODEL {name}')
    targdir='mine_01_{}'
    to_vis = [2, 9, 15, 373, 396, 456, 562, 567, 581, 637, 651, 710, 742, 760, 910, 990]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(join(basedir,name),targdir = targdir,choose_vids=to_vis,device=device)
    print(f'USING MODEL {name}\n\n\n ')
