import torch
import logging
import os
import numpy as np
import fnmatch
from PIL import Image
from libs.utils import batch_project
from scipy.io import loadmat, savemat
from torch.utils.data import Dataset
import argparse
from yacs.config import CfgNode as CN
import torchvision
from torchvision.transforms import functional as F
import json

def get_objid(obj):
    obj_dict = {'ape':1, 'benchvise':2, 'cam':4, 'can':5, 'cat':6, 'driller':8, 'duck':9, 'eggbox':10, 'glue':11, 'holepuncher':12, 
    'iron':13, 'lamp':14, 'phone':15}
    return obj_dict[obj]

def get_lmo_PM_gt_img_list(root, objid):
    PM_file = root + '/{:06d}/scene_gt.json'.format(2)
    with open(PM_file) as f:
        PMs = json.load(f) 
    len_i = len(PMs)
    PM = torch.zeros(len_i, 3, 4)
    img_list = []
    for idx in range(len_i):
        list_idx = PMs[str(idx)]
        objid_list = [temp['obj_id'] for temp in list_idx]
        if objid in objid_list:
            ttt = [ temp for temp in list_idx if temp['obj_id']==objid]
            R = torch.tensor(ttt[0]['cam_R_m2c']).view(1,3,3)
            T = 0.1*torch.tensor(ttt[0]['cam_t_m2c']).view(1,3,1)
            PM[idx,:,:] = torch.cat((R,T),dim=-1)
            img_list.append(idx)
    return PM, img_list

def get_K():
    fx = 572.41140
    fy = 573.57043
    u = 325.26110
    v = 242.04899
    K = torch.tensor(
        [[fx, 0, u],
         [0, fy, v],
         [0, 0, 1]],
        dtype=torch.float)
    return K

class lmo(Dataset):
    def __init__(self, cfg):
        self.img_path = os.path.join(cfg.LMO_DIR,'000002/rgb')
        self.objid = get_objid(cfg.obj)
        self.pts3d = torch.tensor(loadmat('dataset/fps/lm/obj{:02d}_fps128.mat'.format(self.objid))['fps'])[:cfg.N_PTS,:]
        self.npts = cfg.N_PTS
        self.PMs, self.img_list = get_lmo_PM_gt_img_list(cfg.LMO_DIR, self.objid)
        self.cfg = cfg
        self.K = get_K()
        

    def __len__(self,): 
        return 1214 # return the full set, then must create subset using self.img_list

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_path, '{:06d}.png'.format(idx)))
        PM = self.PMs[idx].view(1,3,4)
        pts2d = batch_project(PM,self.pts3d,self.K,angle_axis=False).squeeze()

        W,H = self.cfg.MODEL.IMAGE_SIZE
        xmin = pts2d[:,0].min()-5
        xmax = pts2d[:,0].max()+5
        ymin = pts2d[:,1].min()-5
        ymax = pts2d[:,1].max()+5

        num_objs = 1
        boxes = [xmin, ymin, xmax, ymax]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).view(1,4)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        vis = torch.ones(self.npts, 1)
        vis[pts2d[:,0]<0, 0] = 0
        vis[pts2d[:,0]>W, 0] = 0
        vis[pts2d[:,1]<0, 0] = 0
        vis[pts2d[:,1]>H, 0] = 0
        keypoints = torch.cat((pts2d, vis),dim=-1).view(self.npts, -1, 3)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["keypoints"] = keypoints
        target["PM"] = PM.squeeze()

        img = F.to_tensor(img)

        return img, target












