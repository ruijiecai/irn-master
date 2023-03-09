import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
import sys
sys.path.append(r'/media/crj/irn-master/irn-master')
from my_dataloader import dataloader
from misc import torchutils, imutils

all_image_path=[]

base_path='/media/crj/irn-master/irn-master/IO/Dataset/data/新冠肺炎/Image'
for jpg_folder in os.listdir(base_path):
    for jpg_file in os.listdir(os.path.join(base_path,jpg_folder)):
        image_name=os.path.join(base_path,jpg_folder,jpg_file)
        all_image_path.append(image_name)

base_path='/media/crj/irn-master/irn-master/IO/Dataset/data/普通肺炎/Image'
for jpg_folder in os.listdir(base_path):
    for jpg_file in os.listdir(os.path.join(base_path,jpg_folder)):
        image_name=os.path.join(base_path,jpg_folder,jpg_file)
        all_image_path.append(image_name)

import random
random.shuffle(all_image_path)

from sklearn.model_selection import train_test_split
train_img_path, test_img_path = train_test_split(all_image_path, test_size=0.3)

cudnn.enabled = True

def _work(process_id, model, dataset, args):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(data_loader):

            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            outputs = [model(img[0].cuda(non_blocking=True))
                       for img in pack['img']]

            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                 in outputs]), 0)

            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]

            valid_cat = torch.nonzero(label)[:, 0]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            all_xinguan_image_path[0].split('/media/crj/irn-master/irn-master/IO/Dataset/data/')[1].split('.')[0]
            # save cams
            np.save(os.path.join(args.cam_out_dir, img_name.split() + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model.load_state_dict(torch.load(args.cam_weights_name), strict=True)
    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = dataloader.VOC12ClassificationDatasetMSF(train_img_path ,
                                                             voc12_root=args.voc12_root, scales=args.cam_scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()