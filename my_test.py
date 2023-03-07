import os
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sys
sys.path.append(r'/media/crj/irn-master/irn-master')
sys.path.append(r'/media/crj/irn-master')
import numpy as np
import torch
from torch.utils.data import Dataset
import os.path
import imageio
from misc import imutils
import importlib

# import voc12.dataloader
from misc import pyutils, torchutils

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

CAT_LIST=['新冠肺炎','普通肺炎']
CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))
cls_labels_dict={}
for item in all_image_path:
    cat_name=item.split('/')[8]

    label=np.zeros(len(CAT_LIST))
    label[CAT_NAME_TO_NUM[cat_name]]=1.0
    cls_labels_dict[item]=label

# import torch.nn as nn
# def binary_focal_loss(pred, truth, gamma=2., alpha=.25):
#     eps = 1e-8
#     pred = nn.Softmax(1)(pred)
#     truth = F.one_hot(truth.long(), num_classes = pred.shape[1]).permute(0,3,1,2).contiguous()

#     pt_1 = torch.where(truth == 1, pred, torch.ones_like(pred))
#     pt_0 = torch.where(truth == 0, pred, torch.zeros_like(pred))

#     pt_1 = torch.clamp(pt_1, eps, 1. - eps)
#     pt_0 = torch.clamp(pt_0, eps, 1. - eps)

#     out1 = -torch.mean(alpha * torch.pow(1. - pt_1, gamma) * torch.log(pt_1)) 
#     out0 = -torch.mean((1 - alpha) * torch.pow(pt_0, gamma) * torch.log(1. - pt_0))

#     return out1 + out0

class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img

class VOC12ImageDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None, to_torch=True):

        self.img_name_list = img_name_list_path
        self.voc12_root = voc12_root

        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str =name

        img = np.asarray(imageio.imread(name))

        if self.resize_long:
            img = imutils.random_resize_long(img, self.resize_long[0], self.resize_long[1])

        if self.rescale:
            img = imutils.random_scale(img, scale_range=self.rescale, order=3)

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img = imutils.random_lr_flip(img)

        if self.crop_size:
            if self.crop_method == "random":
                img = imutils.random_crop(img, self.crop_size, 0)
            else:
                img = imutils.top_left_crop(img, self.crop_size, 0)

        if self.to_torch:
            img = imutils.HWC_to_CHW(img)

        return {'name': name_str, 'img': img}

def load_image_label_list_from_npy(img_name_list):

    return np.array([cls_labels_dict[img_name] for img_name in img_name_list])

class VOC12ClassificationDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None):
        super().__init__(img_name_list_path, voc12_root,
                 resize_long, rescale, img_normal, hor_flip,
                 crop_size, crop_method)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        out['label'] = torch.from_numpy(self.label_list[idx])

        return out

from torchvision.ops import sigmoid_focal_loss

def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']

            label = pack['label'].cuda(non_blocking=True)

            x = model(img)
            loss1 = sigmoid_focal_loss(x, label,reduction='mean')

            val_loss_meter.add({'loss1': loss1.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss1')))

    return

def run(args):

    model = getattr(importlib.import_module(args.cam_network), 'Net')()


    train_dataset = VOC12ClassificationDataset(train_img_path, voc12_root=args.voc12_root,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

    val_dataset = VOC12ClassificationDataset(test_img_path, voc12_root=args.voc12_root,
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    for ep in range(args.cam_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))

        for step, pack in enumerate(train_data_loader):

            img = pack['img']
            label = pack['label'].cuda(non_blocking=True)

            x = model(img)
            #loss = F.multilabel_soft_margin_loss(x, label)
            loss=sigmoid_focal_loss(x,label,reduction='mean')
            avg_meter.add({'loss1': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step-1)%100 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss1')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)
                
        else:
            validate(model, val_data_loader)
            timer.reset_stage() 
            
    torch.save(model.module.state_dict(), args.cam_weights_name)
    torch.cuda.empty_cache()