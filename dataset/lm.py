import torch
import logging
import os
import numpy as np
import fnmatch
from PIL import Image
from libs.utils import batch_project
from scipy.io import loadmat, savemat
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
import imgaug as ia
from imgaug.augmentables import Keypoint, KeypointsOnImage
import argparse
from yacs.config import CfgNode as CN
import torchvision
import imageio
import json

def get_objid(obj):
    obj_dict = {'ape':1, 'benchvise':2, 'cam':4, 'can':5, 'cat':6, 'driller':8, 'duck':9, 'eggbox':10, 'glue':11, 'holepuncher':12, 
    'iron':13, 'lamp':14, 'phone':15}
    return obj_dict[obj]

def get_lm_PM_gt(root, objid):
    PM_file = root + '/{:06d}/scene_gt.json'.format(objid)
    with open(PM_file) as f:
        PM = json.load(f) 
    return PM

def get_lm_synt_PM_gt(root, objid):
    PM_file = root + '/{:06d}/scene_gt.json'.format(objid)
    with open(PM_file) as f:
        PM = json.load(f)
    return PM

def get_K(device):
    fx = 572.41140
    fy = 573.57043
    u = 325.26110
    v = 242.04899
    K = torch.tensor(
        [[fx, 0, u],
         [0, fy, v],
         [0, 0, 1]],
        device=device, dtype=torch.float)
    return K

def divide_box(bbox, n_range=(3,6), p_range=(0.25, 0.7), img_w=640, img_h=480):
    # bbox: size [4], format [x,y,w,h]
    n = torch.randint(n_range[0], n_range[1], (1,)).item()
    p = (p_range[1]-p_range[0])*torch.rand(1).item()+p_range[0]
    cells = torch.zeros(n, n, 2)
    occlude = torch.rand(n,n)<=p
    X = bbox[0]
    Y = bbox[1]
    W = bbox[2]
    H = bbox[3]
    if W%n != 0:
        W = W - W%n
    if H%n != 0:
        H = H - H%n
    assert W%n == 0
    assert H%n == 0
    assert X+W <= img_w, 'X: {}, W: {}, img_w: {}'.format(X, W, img_w)
    assert Y+H <= img_h, 'Y: {}, H: {}, img_h: {}'.format(Y, H, img_h)
    w = int(W/n)
    h = int(H/n)
    for i in range(n):
        for j in range(n):
            cells[i,j,0] = X + i*w
            cells[i,j,1] = Y + j*h
    return cells.view(-1,2).long(), occlude.view(-1), w, h

def get_patch_xy(num_patches, img_w, img_h, obj_bbox, cell_w, cell_h):
    patch_xy = torch.zeros(num_patches, 2)
    max_w = img_w - cell_w
    max_h = img_h - cell_h
    X = obj_bbox[0]
    Y = obj_bbox[1]
    XX = X + obj_bbox[2]
    YY = Y + obj_bbox[3]
    assert XX>X and X>=0 and XX<=img_w, 'X {}, XX {}, Y {}, YY {}, cell_w {}, cell_h {}, img_w {}, img_h {}.'.format(X, XX, Y, YY, cell_w, cell_h, img_w, img_h)
    assert YY>Y and Y>=0 and YY<=img_h, 'X {}, XX {}, Y {}, YY {}, cell_w {}, cell_h {}, img_w {}, img_h {}.'.format(X, XX, Y, YY, cell_w, cell_h, img_w, img_h)
    for i in range(num_patches):
        x = torch.randint(0, max_w-1, (1,))
        y = torch.randint(0, max_h-1, (1,))
        trial = 0
        while x>=X and x<XX and y>=Y and y<YY:
            x = torch.randint(0, max_w-1, (1,))
            y = torch.randint(0, max_h-1, (1,))
            trial += 1
            if trial > 1000:
                print('Can find patch! X {}, XX {}, Y {}, YY {}, cell_w {}, cell_h {}, img_w {}, img_h {}.'
                    .format(X, XX, Y, YY, cell_w, cell_h, img_w, img_h))
        patch_xy[i,0] = x
        patch_xy[i,1] = y
    return patch_xy

def get_bbox(pts2d, img_size, coco_format=False):
    W = img_size[-2]
    H = img_size[-1]
    xmin = int(max(pts2d[:,0].min().round().item()-15, 0))
    xmax = int(min(pts2d[:,0].max().round().item()+15, W))
    assert xmax>xmin
    ymin = int(max(pts2d[:,1].min().round().item()-15, 0))
    ymax = int(min(pts2d[:,1].max().round().item()+15, H))
    assert ymax>ymin
    if coco_format:
        return [xmin, ymin, xmax, ymax]
    else:
        return [xmin, ymin, xmax-xmin, ymax-ymin]

def check_if_inside(pts2d, x1, x2, y1, y2):
    r1 = pts2d[:, 0]-0.5 >= x1 -0.5
    r2 = pts2d[:, 0]-0.5 <= x2 -1 + 0.5
    r3 = pts2d[:, 1]-0.5 >= y1 -0.5
    r4 = pts2d[:, 1]-0.5 <= y2 -1 + 0.5
    return r1*r2*r3*r4

def obj_out_of_view(W, H, pts2d):
    xmin = pts2d[:,0].min().item()
    xmax = pts2d[:,0].max().item()
    ymin = pts2d[:,1].min().item()
    ymax = pts2d[:,1].max().item()
    if xmin>W or xmax<0 or ymin>H or ymax<0:
        return True
    else:
        return False

def occlude_obj(img, pts2d, vis=None, p_white_noise=0.1, p_occlude=(0.25, 0.7)):
    # img: image tensor of size [3, h, w]
    _, img_h, img_w = img.size()

    if obj_out_of_view(img_w, img_h, pts2d):
        return img, None

    bbox = get_bbox(pts2d, [img_w, img_h])
    cells, occ_cell, cell_w, cell_h = divide_box(bbox, p_range=p_occlude, img_w=img_w, img_h=img_h)
    num_cells = cells.size(0)
    noise_occ_id = torch.rand(num_cells) <= p_white_noise
    actual_noise_occ = noise_occ_id * occ_cell
    num_patch_occ = occ_cell.sum() - actual_noise_occ.sum()
    patches_xy = get_patch_xy(num_patch_occ, img_w, img_h, bbox, cell_w, cell_h)
    j = 0
    for i in range(num_cells):
        if occ_cell[i]:
            x1 = cells[i,0].item()
            x2 = x1 + cell_w
            y1 = cells[i,1].item()
            y2 = y1 + cell_h

            if vis is not None:
                vis = vis*(~check_if_inside(pts2d, x1, x2, y1, y2))

            if noise_occ_id[i]: # white_noise occlude
                img[:, y1:y2, x1:x2] = torch.rand(3, cell_h, cell_w) 
            else: # patch occlude
                xx1 = patches_xy[j, 0].long().item()
                xx2 = xx1 + cell_w
                yy1 = patches_xy[j, 1].long().item()
                yy2 = yy1 + cell_h
                img[:, y1:y2, x1:x2] = img[:, yy1:yy2, xx1:xx2].clone()
                j += 1
    assert num_patch_occ == j
    return img, vis


def kps2tensor(kps):
    n = len(kps.keypoints)
    pts2d = np.array([kps.keypoints[i].coords for i in range(n)])
    return torch.tensor(pts2d, dtype=torch.float).squeeze()


def augment_lm(img, pts2d, device, is_synt=False):
    assert len(img.size()) == 3

    H, W = img.size()[-2:]
    bbox = get_bbox(pts2d, [W, H])
    min_x_shift = int(-bbox[0]+10)
    max_x_shift = int(W-bbox[0]-bbox[2]-10)
    min_y_shift = int(-bbox[1]+10)
    max_y_shift = int(H-bbox[1]-bbox[3]-10)
    assert max_x_shift > min_x_shift, 'H: {}, W: {}, bbox: {}, {}, {}, {}'.format(H, W, bbox[0], bbox[1], bbox[2], bbox[3])
    assert max_y_shift > min_y_shift, 'H: {}, W: {}, bbox: {}, {}, {}, {}'.format(H, W, bbox[0], bbox[1], bbox[2], bbox[3])

    img = img.permute(1,2,0).numpy()
    nkp = pts2d.size(0)
    kp_list = [Keypoint(x=pts2d[i][0].item(), y=pts2d[i][1].item()) for i in range(nkp)] 
    kps = KeypointsOnImage(kp_list, shape=img.shape)

    if is_synt:
        step0 = iaa.Affine(scale=(0.35,0.6))
        img, kps = step0(image=img, keypoints=kps)

    rotate = iaa.Affine(rotate=(-30, 30))
    scale = iaa.Affine(scale=(0.8, 1.2))
    trans = iaa.Affine(translate_px={"x": (min_x_shift, max_x_shift), "y": (min_y_shift, max_y_shift)})
    bright = iaa.MultiplyAndAddToBrightness(mul=(0.7, 1.3))
    hue_satu = iaa.MultiplyHueAndSaturation(mul_hue=(0.95,1.05), mul_saturation=(0.5,1.5))
    contrast = iaa.GammaContrast((0.8, 1.2))
    random_aug = iaa.SomeOf((3, 6), [rotate, trans, scale, bright, hue_satu, contrast])
    img1, kps1 = random_aug(image=img, keypoints=kps)

    img1 = torch.tensor(img1).permute(2,0,1).to(device)
    pts2d1 = kps2tensor(kps1).to(device)

    if pts2d1[:,0].min()>W or pts2d1[:,0].max()<0 or pts2d1[:,1].min()>H or pts2d1[:,1].max()<0:
        img1 = torch.tensor(img).permute(2,0,1).to(device)
        pts2d1 = kps2tensor(kps).to(device)

    return img1, img1.clone(), pts2d1


def blackout(img, pts2d):
    assert len(img.size()) == 3
    H, W = img.size()[-2:]
    x, y, w, h = get_bbox(pts2d, [W, H])
    img2 = torch.zeros_like(img)
    img2[:, y:y+h, x:x+w] = img[:, y:y+h, x:x+w].clone()
    return img2


class lm(Dataset):
    def __init__(self, cfg):
        self.objid = get_objid(cfg.obj)
        self.lm_root = cfg.LM_DIR
        self.data_path = os.path.join(self.lm_root,'{:06d}/rgb'.format(self.objid))
        self.PMs = get_lm_PM_gt(self.lm_root, self.objid)
        self.pts3d = torch.tensor(loadmat('dataset/fps/lm/obj{:02d}_fps128.mat'.format(self.objid))['fps'])[:cfg.N_PTS,:]
        self.npts = cfg.N_PTS
        self.cfg = cfg
        self.K = get_K('cpu')
        self.n_lm = len(self.PMs)

    def __len__(self,):
        return self.n_lm

    def __getitem__(self, idx):
        img = imageio.imread(os.path.join(self.data_path, '{:06d}.png'.format(idx)))
        img = torch.tensor(img).permute(2,0,1)
        R = torch.tensor(self.PMs[str(idx)][0]['cam_R_m2c']).view(1,3,3)
        T = 0.1*torch.tensor(self.PMs[str(idx)][0]['cam_t_m2c']).view(1,3,1)
        PM = torch.cat((R,T),dim=-1)
        pts2d = batch_project(PM, self.pts3d, self.K, angle_axis=False).squeeze()

        tru = torch.ones(1, dtype=torch.bool)
        fal = torch.zeros(1, dtype=torch.bool)
        num_objs = 1
        W, H = self.cfg.MODEL.IMAGE_SIZE

        boxes = get_bbox(pts2d, self.cfg.MODEL.IMAGE_SIZE, coco_format=True)
        boxes = torch.as_tensor(boxes, dtype=torch.float32).view(1,4)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        vis = torch.ones(self.npts, 1)
        vis[pts2d[:,0]<0, 0] = 0
        vis[pts2d[:,0]>W, 0] = 0
        vis[pts2d[:,1]<0, 0] = 0
        vis[pts2d[:,1]>H, 0] = 0
        keypoints = torch.cat((pts2d, vis),dim=-1).view(1, self.npts, 3)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes.clone()
        target["labels"] = labels.clone()
        target["image_id"] = image_id.clone()
        target["area"] = area.clone()
        target["iscrowd"] = iscrowd.clone()
        target["keypoints"] = keypoints.clone()
        target['fmco1'] = tru.clone()
        target['fmco2'] = fal.clone()

        return img.float()/255, target


class lm_with_synt(Dataset):
    def __init__(self, cfg):
        self.objid = get_objid(cfg.obj)
        self.synt_root = cfg.LM_SYNT_DIR
        self.lm_root = cfg.LM_DIR
        self.data_path = os.path.join(self.lm_root,'{:06d}/rgb'.format(self.objid))
        self.PMs = get_lm_PM_gt(self.lm_root, self.objid)
        self.PMs_synt = get_lm_synt_PM_gt(self.synt_root, self.objid)
        self.pts3d = torch.tensor(loadmat('dataset/fps/lm/obj{:02d}_fps128.mat'.format(self.objid))['fps'])[:cfg.N_PTS,:]
        self.npts = cfg.N_PTS
        self.cfg = cfg
        self.K = get_K('cpu')
        self.n_lm = len(self.PMs)
        self.n_synt = len(self.PMs_synt)

    def __len__(self,):
        return self.n_lm + self.n_synt

    def __getitem__(self, idx):
        idx0 = idx
        if idx < self.n_lm:
            img = imageio.imread(os.path.join(self.data_path, '{:06d}.png'.format(idx)))
            img = torch.tensor(img).permute(2,0,1)
            R = torch.tensor(self.PMs[str(idx)][0]['cam_R_m2c']).view(1,3,3)
            T = 0.1*torch.tensor(self.PMs[str(idx)][0]['cam_t_m2c']).view(1,3,1)
            PM = torch.cat((R,T),dim=-1)
            pts2d = batch_project(PM, self.pts3d, self.K, angle_axis=False).squeeze()
            is_synt = False
        else:
            idx = idx-self.n_lm
            img = imageio.imread(os.path.join(self.synt_root, '{:06d}/rgb/{:06d}.png'.format(self.objid, idx)))
            img = torch.tensor(img).permute(2,0,1)
            R = torch.tensor(self.PMs_synt[str(idx)][0]['cam_R_m2c']).view(1,3,3)
            T = 0.1*torch.tensor(self.PMs_synt[str(idx)][0]['cam_t_m2c']).view(1,3,1)
            PM = torch.cat((R,T),dim=-1)
            pts2d = batch_project(PM, self.pts3d, self.K, angle_axis=False).squeeze()
            is_synt = True

        img1, img2, pts2d = augment_lm(img, pts2d, 'cpu', is_synt)
        if torch.rand(1) < 0.95:
            img2, _ = occlude_obj(img2.clone(), pts2d.clone(), p_occlude=(0.15, 0.7))
        img2 = blackout(img2, pts2d.clone())

        num_objs = 1
        W, H = self.cfg.MODEL.IMAGE_SIZE

        boxes = get_bbox(pts2d, self.cfg.MODEL.IMAGE_SIZE, coco_format=True)
        boxes = torch.as_tensor(boxes, dtype=torch.float32).view(1,4)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        vis = torch.ones(self.npts, 1)
        vis[pts2d[:,0]<0, 0] = 0
        vis[pts2d[:,0]>W, 0] = 0
        vis[pts2d[:,1]<0, 0] = 0
        vis[pts2d[:,1]>H, 0] = 0
        keypoints = torch.cat((pts2d, vis),dim=-1).view(1, self.npts, 3)

        image_id = torch.tensor([idx0])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target1 = {}
        target1["boxes"] = boxes.clone()
        target1["labels"] = labels.clone()
        target1["image_id"] = image_id.clone()
        target1["area"] = area.clone()
        target1["iscrowd"] = iscrowd.clone()
        target1["keypoints"] = keypoints.clone()

        target2 = {}
        target2["boxes"] = boxes.clone()
        target2["labels"] = labels.clone()
        target2["image_id"] = image_id.clone()
        target2["area"] = area.clone()
        target2["iscrowd"] = iscrowd.clone()
        target2["keypoints"] = keypoints.clone()

        return img1.float()/255, img2.float()/255, target1, target2
















