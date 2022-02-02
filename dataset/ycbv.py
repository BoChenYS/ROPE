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

def get_ycbv_objid(obj):
    # obj_dict = {'master_chef_can':1, 'cracker_box':2, 'sugar_box':3, 'tomato_soup_can':4, 'mustard_bottle':5, 'tuna_fish_can':6, 'pudding_box':7, 'gelatin_box':8, 
    # 'potted_meat_can':9, 'banana':10, 'pitcher_base':11, 'bleach_cleanser':12, 'bowl':13, 'mug':14, 'power_drill':15, 'wood_block':16, 'scissors':17, 'large_marker':18, 
    # 'large_clamp':19, 'extra_large_clamp':20, 'foam_brick':21}
    obj_dict = {'01':1, '02':2, '03':3, '04':4, '05':5, '06':6, '07':7, '08':8, '09':9, '10':10, '11':11, '12':12, '13':13, '14':14, '15':15, '16':16, '17':17, '18':18,
    '19':19, '20':20, '21':21}
    return obj_dict[obj]

def gen_ycbv_models_from_ply(root, obj_id):
    # get the 3D mesh of an object
    device = 'cpu'
    file = root+'/models/obj_{:06d}.ply'.format(obj_id)
    f = open(file)
    f.readline()
    line = f.readline()
    sss = line.strip()

    while sss != 'end_header':
        if line.split()[1] == 'vertex':
            num_vertex = line.split()[2]
        if line.split()[1] == 'face':
            num_face = line.split()[2]
        line = f.readline()
        sss = line.strip()

    pts3d_mesh = torch.zeros(np.int(num_vertex), 3, device=device)
    for N in range(np.int(num_vertex)):
        line = f.readline() 
        pts3d_mesh[N, :] = 0.1*torch.tensor(np.float32(line.split()[:3]), device=device).view(1, 3) # times 0.1 to convert mm to cm

    triangle_ids = torch.zeros(np.int(num_face), 3).long()
    for N in range(np.int(num_face)):
        line = f.readline()
        triangle_ids[N, :] = torch.tensor(np.int64(line.split()[1:4]))

    savemat(root+'/models/obj{:02d}.mat'.format(obj_id), {'pts3d':pts3d_mesh.numpy(), 'triangle_ids':triangle_ids.numpy()})

def find_ycbv_train_seq_has_obj(root, objid):
    all_seq = list(range(0,48))+list(range(60, 92))
    obj_seq = []
    for i in all_seq:
        anno_file = root + '/train_real/{:06d}/scene_gt.json'.format(i)
        with open(anno_file) as f:
            anno = json.load(f) 
        obj_list = anno['1']
        n_obj = len(obj_list)
        for j in range(n_obj):
            if obj_list[j]['obj_id']==objid:
                obj_seq.append(i)
    return obj_seq

def find_ycbv_test_seq_has_obj(root, objid):
    all_seq = list(range(48,60))
    obj_seq = []
    for i in all_seq:
        anno_file = root + '/test/{:06d}/scene_gt.json'.format(i)
        with open(anno_file) as f:
            anno = json.load(f) 
        obj_list = anno['1']
        n_obj = len(obj_list)
        for j in range(n_obj):
            if obj_list[j]['obj_id']==objid:
                obj_seq.append(i)
    return obj_seq

def gen_ycbv_train_annos(root, objid):
    obj_seq = find_ycbv_train_seq_has_obj(root, objid)
    seq_ids = []
    img_ids = []
    PMs = torch.zeros(0,3,4)
    Ks = torch.zeros(0,3,3)
    for seq in obj_seq:
        anno_file = root + '/train_real/{:06d}/scene_gt.json'.format(seq)
        cam_file = root + '/train_real/{:06d}/scene_camera.json'.format(seq)
        info_file = root + '/train_real/{:06d}/scene_gt_info.json'.format(seq)

        with open(anno_file) as f:
            anno = json.load(f) 
        n_imgs = len(anno)
        with open(cam_file) as f:
            cam = json.load(f) 
        with open(info_file) as f:
            info = json.load(f) 
        assert len(cam) == n_imgs
        assert len(info) == n_imgs

        anno_list = anno['1']
        n_obj = len(anno_list)
        obj_pos = [i for i in range(n_obj) if anno_list[i]['obj_id']==objid][0]

        cam_dict = cam['1']
        K = torch.tensor(cam_dict['cam_K']).view(1, 3,3)

        for i in range(n_imgs):
            img_id = i+1
            vis_frac = info[str(img_id)][obj_pos]['visib_fract']
            if vis_frac > 0.3:
                R = torch.tensor(anno[str(img_id)][obj_pos]['cam_R_m2c']).view(1,3,3)
                T = 0.1*torch.tensor(anno[str(img_id)][obj_pos]['cam_t_m2c']).view(1,3,1)
                PM = torch.cat((R,T),dim=-1)
                PMs = torch.cat((PMs, PM), dim=0)
                Ks = torch.cat((Ks, K), dim=0)
                seq_ids.append(seq)
                img_ids.append(img_id)
    savemat(root+'/train_annos/obj{:02d}.mat'.format(objid), {'PMs':PMs.numpy(), 'Ks':Ks.numpy(), 'seq_ids':seq_ids, 'img_ids':img_ids})

def gen_ycbv_train_synt_annos(root, objid):
    obj_seq = list(range(80))
    seq_ids = []
    img_ids = []
    PMs = torch.zeros(0,3,4)
    Ks = torch.zeros(0,3,3)
    for seq in obj_seq:
        anno_file = root + '/train_synt/{:06d}/scene_gt.json'.format(seq)
        cam_file = root + '/train_synt/{:06d}/scene_camera.json'.format(seq)
        info_file = root + '/train_synt/{:06d}/scene_gt_info.json'.format(seq)

        with open(anno_file) as f:
            anno = json.load(f) 
        n_imgs = len(anno)
        with open(cam_file) as f:
            cam = json.load(f) 
        with open(info_file) as f:
            info = json.load(f) 
        assert n_imgs == 1000
        assert len(cam) == n_imgs
        assert len(info) == n_imgs

        for i in range(n_imgs):
            img_id = i+1000*seq

            anno_list = anno[str(img_id)]
            n_obj = len(anno_list)
            obj_pos_list = [ii for ii in range(n_obj) if anno_list[ii]['obj_id']==objid]
            if len(obj_pos_list)==0:
                continue
            else:
                obj_pos = obj_pos_list[0]

            cam_dict = cam[str(img_id)]
            K = torch.tensor(cam_dict['cam_K']).view(1, 3,3)

            vis_frac = info[str(img_id)][obj_pos]['visib_fract']
            if vis_frac > 0.3:
                R = torch.tensor(anno_list[obj_pos]['cam_R_m2c']).view(1,3,3)
                T = 0.1*torch.tensor(anno_list[obj_pos]['cam_t_m2c']).view(1,3,1)
                PM = torch.cat((R,T),dim=-1)
                PMs = torch.cat((PMs, PM), dim=0)
                Ks = torch.cat((Ks, K), dim=0)
                seq_ids.append(seq)
                img_ids.append(img_id)
    savemat(root+'/train_annos/obj{:02d}_synt.mat'.format(objid), {'PMs':PMs.numpy(), 'Ks':Ks.numpy(), 'seq_ids':seq_ids, 'img_ids':img_ids})

def gen_ycbv_test_annos(root, objid):
    obj_seq = find_ycbv_test_seq_has_obj(root, objid)
    seq_ids = []
    img_ids = []
    PMs = torch.zeros(0,3,4)
    Ks = torch.zeros(0,3,3)
    keyframe = loadmat(root+'/test_annos/keyframe.mat')
    for seq in obj_seq:
        anno_file = root + '/test/{:06d}/scene_gt.json'.format(seq)
        with open(anno_file) as f:
            anno = json.load(f) 
        img_list = keyframe['seq{}'.format(seq)].squeeze()
        anno_list = anno['1']
        n_obj = len(anno_list)
        obj_pos = [i for i in range(n_obj) if anno_list[i]['obj_id']==objid][0]

        cam_file = root + '/test/{:06d}/scene_camera.json'.format(seq)
        with open(cam_file) as f:
            cam = json.load(f)
        cam_dict = cam['1']
        K = torch.tensor(cam_dict['cam_K']).view(1,3,3)

        for img_id in img_list:
            R = torch.tensor(anno[str(img_id)][obj_pos]['cam_R_m2c']).view(1,3,3)
            T = 0.1*torch.tensor(anno[str(img_id)][obj_pos]['cam_t_m2c']).view(1,3,1)
            PM = torch.cat((R,T),dim=-1)
            PMs = torch.cat((PMs, PM), dim=0)
            Ks = torch.cat((Ks, K), dim=0)
            seq_ids.append(seq)
            img_ids.append(img_id)
    savemat(root+'/test_annos/obj{:02d}.mat'.format(objid), {'PMs':PMs.numpy(), 'Ks':Ks.numpy(), 'seq_ids':seq_ids, 'img_ids':img_ids})


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
                print('Warning. Cannot find patch outside bbox, using patch overlapping bbox. X {}, XX {}, Y {}, YY {}, cell_w {}, cell_h {}, img_w {}, img_h {}.'
                    .format(X, XX, Y, YY, cell_w, cell_h, img_w, img_h))
                break
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


def aug_ycbv(img, pts2d, device):
    assert len(img.size()) == 3

    img0 = img.clone()
    pts2d0 = pts2d.clone()

    H, W = img.size()[-2:]
    bbox = get_bbox(pts2d, [W, H])
    min_x_shift = int(-bbox[0])
    max_x_shift = int(W-bbox[0]-bbox[2])
    min_y_shift = int(-bbox[1])
    max_y_shift = int(H-bbox[1]-bbox[3])
    assert max_x_shift >= min_x_shift, 'H: {}, W: {}, bbox: {}, {}, {}, {}'.format(H, W, bbox[0], bbox[1], bbox[2], bbox[3])
    assert max_y_shift >= min_y_shift, 'H: {}, W: {}, bbox: {}, {}, {}, {}'.format(H, W, bbox[0], bbox[1], bbox[2], bbox[3])

    img = img.permute(1,2,0).numpy()
    nkp = pts2d.size(0)
    kp_list = [Keypoint(x=pts2d[i][0].item(), y=pts2d[i][1].item()) for i in range(nkp)] 
    kps = KeypointsOnImage(kp_list, shape=img.shape)

    rotate = iaa.Affine(rotate=(-30, 30))
    scale = iaa.Affine(scale=(0.8, 1.2))
    trans = iaa.Affine(translate_px={"x": (min_x_shift, max_x_shift), "y": (min_y_shift, max_y_shift)})
    bright = iaa.MultiplyAndAddToBrightness(mul=(0.7, 1.3))
    hue_satu = iaa.MultiplyHueAndSaturation(mul_hue=(0.95,1.05), mul_saturation=(0.5,1.5))
    contrast = iaa.GammaContrast((0.8, 1.2))
    random_aug = iaa.SomeOf((3, 6), [rotate, trans, scale, bright, hue_satu, contrast])

    if torch.rand(1) < 0.95:
        img, kps = random_aug(image=img, keypoints=kps)

    img = torch.tensor(img).permute(2,0,1).to(device)
    pts2d = kps2tensor(kps).to(device)

    if obj_out_of_view(W, H, pts2d):
        return img0, img0.clone(), pts2d0
    else:
        return img, img.clone(), pts2d

def blackout(img, pts2d):
    assert len(img.size()) == 3
    H, W = img.size()[-2:]
    x, y, w, h = get_bbox(pts2d, [W, H])
    img2 = torch.zeros_like(img)
    img2[:, y:y+h, x:x+w] = img[:, y:y+h, x:x+w].clone()
    return img2


class ycbv_train_w_synt(Dataset):
    def __init__(self, cfg):
        self.objid = get_ycbv_objid(cfg.obj)
        self.root = cfg.YCBV_DIR

        self.data_path = os.path.join(self.root, 'train_real')
        self.annos = loadmat(self.root+'/train_annos/obj{:02d}.mat'.format(self.objid))
        self.seq_ids = self.annos['seq_ids'].squeeze()
        self.img_ids = self.annos['img_ids'].squeeze()
        self.Ks = self.annos['Ks']
        self.PMs = self.annos['PMs']

        self.data_path_synt = os.path.join(self.root, 'train_synt')
        self.annos_synt = loadmat(self.root+'/train_annos/obj{:02d}_synt.mat'.format(self.objid))
        self.seq_ids_synt = self.annos_synt['seq_ids'].squeeze()
        self.img_ids_synt = self.annos_synt['img_ids'].squeeze()
        self.Ks_synt = self.annos_synt['Ks']
        self.PMs_synt = self.annos_synt['PMs']

        self.pts3d = torch.tensor(loadmat('dataset/fps/ycbv/obj{:02d}_fps128.mat'.format(self.objid))['fps'])[:cfg.N_PTS,:]
        self.npts = cfg.N_PTS
        self.cfg = cfg
        self.n_real = len(self.img_ids)
        self.n_dataset = len(self.img_ids)+len(self.img_ids_synt)

    def __len__(self,):
        return self.n_dataset

    def __getitem__(self, idx):
        if idx < self.n_real:
            img = imageio.imread(os.path.join(self.data_path, '{:06d}/rgb/{:06d}.png'.format(self.seq_ids[idx], self.img_ids[idx])))
            img = torch.tensor(img).permute(2,0,1)
            PM = torch.tensor(self.PMs[idx]).unsqueeze(0)
            K = torch.tensor(self.Ks[idx])
        else:
            idx2 = idx-self.n_real
            img = imageio.imread(os.path.join(self.data_path_synt, '{:06d}/rgb/{:06d}.png'.format(self.seq_ids_synt[idx2], self.img_ids_synt[idx2])))
            img = torch.tensor(img[:,:,:3]).permute(2,0,1)
            PM = torch.tensor(self.PMs_synt[idx2]).unsqueeze(0)
            K = torch.tensor(self.Ks_synt[idx2])

        pts2d = batch_project(PM, self.pts3d, K, angle_axis=False).squeeze()
        img1, img2, pts2d = aug_ycbv(img, pts2d, 'cpu')
        if torch.rand(1) < 0.95:
            img2, _ = occlude_obj(img2.clone(), pts2d.clone(), p_occlude=(0.1, 0.4))
        img2 = blackout(img2, pts2d.clone())

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


class ycbv_test(Dataset):
    def __init__(self, cfg):
        self.objid = get_ycbv_objid(cfg.obj)
        self.root = cfg.YCBV_DIR
        self.data_path = os.path.join(self.root,'test')
        self.annos = loadmat(self.root+'/test_annos/obj{:02d}.mat'.format(self.objid))
        self.seq_ids = self.annos['seq_ids'].squeeze()
        self.img_ids = self.annos['img_ids'].squeeze()
        self.Ks = self.annos['Ks']
        self.PMs = self.annos['PMs']
        self.pts3d = torch.tensor(loadmat('dataset/fps/ycbv/obj{:02d}_fps128.mat'.format(self.objid))['fps'])[:cfg.N_PTS,:]
        self.npts = cfg.N_PTS
        self.cfg = cfg
        self.n_dataset = len(self.img_ids)

    def __len__(self,):
        return self.n_dataset

    def __getitem__(self, idx):
        img = imageio.imread(os.path.join(self.data_path, '{:06d}/rgb/{:06d}.png'.format(self.seq_ids[idx], self.img_ids[idx])))
        img = torch.tensor(img).permute(2,0,1)
        PM = torch.tensor(self.PMs[idx]).unsqueeze(0)
        K = torch.tensor(self.Ks[idx])
        pts2d = batch_project(PM, self.pts3d, K, angle_axis=False).squeeze()

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

        return img.float()/255, target






























