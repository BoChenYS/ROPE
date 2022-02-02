import numpy as np
import torch
import os
from scipy.io import loadmat, savemat
from sklearn.neighbors import KDTree
import kornia as kn
import logging
import json

def get_logger(cfg):
    if not os.path.exists(cfg.OUTPUT_DIR+'/'+cfg.obj+'/'):
        os.mkdir(cfg.OUTPUT_DIR+'/'+cfg.obj+'/')
    logging.basicConfig(filename=cfg.OUTPUT_DIR+'/'+cfg.obj+'/'+cfg.log_name+'.out', level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %I:%M:%S %p')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    return logger

def get_objid(obj):
    obj_dict = {'ape':1, 'benchvise':2, 'cam':4, 'can':5, 'cat':6, 'driller':8, 'duck':9, 'eggbox':10, 'glue':11, 'holepuncher':12, 
    'iron':13, 'lamp':14, 'phone':15}
    return obj_dict[obj]

def get_ycbv_objid(obj):
    # obj_dict = {'master_chef_can':1, 'cracker_box':2, 'sugar_box':3, 'tomato_soup_can':4, 'mustard_bottle':5, 'tuna_fish_can':6, 'pudding_box':7, 'gelatin_box':8, 
    # 'potted_meat_can':9, 'banana':10, 'pitcher_base':11, 'bleach_cleanser':12, 'bowl':13, 'mug':14, 'power_drill':15, 'wood_block':16, 'scissors':17, 'large_marker':18, 
    # 'large_clamp':19, 'extra_large_clamp':20, 'foam_brick':21}
    obj_dict = {'01':1, '02':2, '03':3, '04':4, '05':5, '06':6, '07':7, '08':8, '09':9, '10':10, '11':11, '12':12, '13':13, '14':14, '15':15, '16':16, '17':17, '18':18,
    '19':19, '20':20, '21':21}
    return obj_dict[obj]

def get_lm_img_idx(cfg, n_lm, n_lm_synt=None):
    objid = get_objid(cfg.obj)
    f = open(os.path.join(cfg.LM_DIR, '{:06d}/training_range.txt'.format(objid)))
    train_idx = [int(x) for x in f]
    whole_idx = list(range(n_lm))
    test_idx = [item for item in whole_idx if item not in train_idx]
    if n_lm_synt is None:
        return train_idx, test_idx
    else:
        synt_idx = list(range(n_lm, n_lm+n_lm_synt))
        train_idx.extend(synt_idx)
        return train_idx, test_idx

def get_lm_dataset_size(cfg):
    objid = get_objid(cfg.obj)
    PM_file = cfg.LM_DIR + '/{:06d}/scene_gt.json'.format(objid)
    with open(PM_file) as f:
        PM = json.load(f) 
    return len(PM)

def get_lm_o_fps(cfg):
    # get the Farthest Point Sample of the 3D mesh
    objid = get_objid(cfg.obj)
    fps_file = cfg.LM_DIR+'/lm_models/lm_fps/obj{:02d}_fps128.mat'.format(objid)
    fpsvis = loadmat(fps_file)
    pts3d_fps = torch.tensor(fpsvis['fps'][0:cfg.N_PTS, :]) # size [N_PTS, 3]
    return pts3d_fps

def get_lm_o_3dmodel(cfg, homo=False):
    objid = get_objid(cfg.obj)
    model_file = cfg.LM_DIR+'/lm_models/lm_meshes_cm/obj{:02d}.mat'.format(objid)
    pts3d = loadmat(model_file)['pts3d']
    pts3d = torch.tensor(pts3d, dtype=torch.float)
    if homo:
        pts3d = torch.cat((pts3d, torch.ones(pts3d.size(0), 1)), dim=-1)
    return pts3d

def get_ycbv_fps(cfg):
    # get the Farthest Point Sample of the 3D mesh
    objid = get_ycbv_objid(cfg.obj)
    fps_file = cfg.YCBV_DIR+'/models/obj{:02d}_fps128.mat'.format(objid)
    fps = loadmat(fps_file)
    pts3d_fps = torch.tensor(fps['fps'][0:cfg.N_PTS, :]) # size [N_PTS, 3]
    return pts3d_fps

def get_ycbv_3dmodel(cfg, homo=False):
    objid = get_ycbv_objid(cfg.obj)
    model_file = cfg.YCBV_DIR+'/models/obj{:02d}.mat'.format(objid)
    model = loadmat(model_file)
    pts3d = torch.tensor(model['pts3d'])
    is_sym = model['sym']
    if homo:
        pts3d = torch.cat((pts3d, torch.ones(pts3d.size(0), 1)), dim=-1)
    return pts3d, is_sym

def get_lm_pose(cfg, idx):
    objid = get_objid(cfg.obj)
    PM_file = cfg.LM_DIR + '/{:06d}/scene_gt.json'.format(objid)
    with open(PM_file) as f:
        PMs = json.load(f) 
    R = torch.tensor(PMs[str(idx)][0]['cam_R_m2c']).view(1,3,3)
    T = 0.1*torch.tensor(PMs[str(idx)][0]['cam_t_m2c']).view(1,3,1)
    return torch.cat((R,T),dim=-1)

def get_lmo_pose(cfg, idx):
    objid = get_objid(cfg.obj)
    PM_file = cfg.LMO_DIR + '/000002/scene_gt.json'
    with open(PM_file) as f:
        PMs = json.load(f) 
    list_idx = PMs[str(idx)]
    objid_list = [temp['obj_id'] for temp in list_idx]
    assert objid in objid_list, 'Image id {} doesn\'t have object {} in sight.'.format(idx, objid)
    ttt = [ temp for temp in list_idx if temp['obj_id']==objid]
    R = torch.tensor(ttt[0]['cam_R_m2c']).view(1,3,3)
    T = 0.1*torch.tensor(ttt[0]['cam_t_m2c']).view(1,3,1)
    return torch.cat((R,T),dim=-1)

def get_PM_gt(cfg, dataset_name, idx):
    if dataset_name == 'lm':
        return get_lm_pose(cfg, idx)
    if dataset_name == 'lmo':
        return get_lmo_pose(cfg, idx)

def get_distance(cfg):
    # get the diameter of the object (in cm)
    objid = get_objid(cfg.obj)
    with open(os.path.join(cfg.LM_DIR,'lm_models/models/models_info.json')) as f:
        models_info = json.load(f)
    diameter = 0.1*torch.tensor(models_info[str(objid)]['diameter']).view(1)
    assert diameter.size()[0] == 1
    return diameter

def get_ycbv_distance(cfg):
    # get the diameter of the object (in cm)
    objid = get_ycbv_objid(cfg.obj)
    with open(os.path.join(cfg.YCBV_DIR,'models/models_info.json')) as f:
        models_info = json.load(f)
    diameter = 0.1*torch.tensor(models_info[str(objid)]['diameter']).view(1)
    assert diameter.size()[0] == 1
    return diameter

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

def get_kp_consensus_aslist(pts2d1, pts2d2, pts2d3, thres, n_min):
    bs = pts2d1.size(0)
    npts = pts2d1.size(1)
    pts2dc = []
    ids = []
    for i in range(bs):
        p1 = pts2d1[i]
        p2 = pts2d2[i]
        p3 = pts2d3[i]

        # dist = (p1-p2).norm(dim=-1) + (p1-p3).norm(dim=-1)
        dist = (p1-p2).norm(dim=-1)
        # dist = (p1-p3).norm(dim=-1)
        
        ids_i = np.where((dist<thres).cpu().numpy())[0]
        if len(ids_i) < n_min:
            ids_i = dist.sort()[1][:n_min]
        pts = p1[ids_i]
        pts2dc.append(pts)
        ids.append(ids_i)
    return pts2dc, ids


def ADD_accuracy(P, pts3d_h, PM_gt, diameter, P_is_matrix=False):
    bs = P.size(0)
    if P_is_matrix:
        PM = P
    else:
        R_out = kn.angle_axis_to_rotation_matrix(P[:, 0:3].view(bs, 3))
        PM = torch.cat((R_out[:, 0:3, 0:3], P[:, 3:6].view(bs, 3, 1)), dim=-1)
    pts3d_cam = pts3d_h.matmul(PM.transpose(1, 2))
    pts3d_cam_gt = pts3d_h.matmul(PM_gt.transpose(1, 2))
    diff = pts3d_cam - pts3d_cam_gt
    mean_dis = diff.norm(p=2,dim=2).mean(dim=1)
    corrects = (mean_dis < 0.1*diameter).float()
    n_correct = corrects.sum().item()
    return n_correct, bs

def ADD_accuracy_withID(P, pts3d_h, PM_gt, diameter, P_is_matrix=False):
    bs = P.size(0)
    if P_is_matrix:
        PM = P
    else:
        R_out = kn.angle_axis_to_rotation_matrix(P[:, 0:3].view(bs, 3))
        PM = torch.cat((R_out[:, 0:3, 0:3], P[:, 3:6].view(bs, 3, 1)), dim=-1)
    pts3d_cam = pts3d_h.matmul(PM.transpose(1, 2))
    pts3d_cam_gt = pts3d_h.matmul(PM_gt.transpose(1, 2))
    diff = pts3d_cam - pts3d_cam_gt
    mean_dis = diff.norm(p=2,dim=2).mean(dim=1)
    corrects = (mean_dis < 0.1*diameter).float()
    n_correct = corrects.sum().item()
    return n_correct, bs, corrects.view(bs, 1)

def ADDS_accuracy(P, pts3d_h, PM_gt, diameter, P_is_matrix=False):
    bs = P.size(0)
    n = pts3d_h.size(0)
    if P_is_matrix:
        PM = P
    else:
        R_out = kn.angle_axis_to_rotation_matrix(P[:, 0:3].view(bs, 3))
        PM = torch.cat((R_out[:, 0:3, 0:3], P[:, 3:6].view(bs, 3, 1)), dim=-1)
    pts3d_cam = pts3d_h.matmul(PM.transpose(1, 2))
    pts3d_cam_gt = pts3d_h.matmul(PM_gt.transpose(1, 2))
    dis = torch.zeros(bs,n,device=pts3d_h.device)
    for i in range(bs):
        kdt = KDTree(pts3d_cam_gt[i].cpu())
        dis[i] = torch.tensor(kdt.query(pts3d_cam[i].detach().cpu(),k=1)[0]).squeeze()
    mean_dis = dis.mean(dim=1)
    corrects = (mean_dis < 0.1*diameter).float()
    n_correct = corrects.sum().item()
    return n_correct, bs

def ADDS_accuracy_withID(P, pts3d_h, PM_gt, diameter, P_is_matrix=False):
    bs = P.size(0)
    n = pts3d_h.size(0)
    if P_is_matrix:
        PM = P
    else:
        R_out = kn.angle_axis_to_rotation_matrix(P[:, 0:3].view(bs, 3))
        PM = torch.cat((R_out[:, 0:3, 0:3], P[:, 3:6].view(bs, 3, 1)), dim=-1)
    pts3d_cam = pts3d_h.matmul(PM.transpose(1, 2))
    pts3d_cam_gt = pts3d_h.matmul(PM_gt.transpose(1, 2))
    dis = torch.zeros(bs,n,device=pts3d_h.device)
    for i in range(bs):
        kdt = KDTree(pts3d_cam_gt[i].cpu())
        dis[i] = torch.tensor(kdt.query(pts3d_cam[i].detach().cpu(),k=1)[0]).squeeze()
    mean_dis = dis.mean(dim=1)
    corrects = (mean_dis < 0.1*diameter).float()
    n_correct = corrects.sum().item()
    return n_correct, bs, corrects.view(bs, 1)

def batch_project(P, pts3d, K, angle_axis=True):
    n = pts3d.size(0)
    bs = P.size(0)
    device = P.device
    pts3d_h = torch.cat((pts3d, torch.ones(n, 1, device=device)), dim=-1)
    if angle_axis:
        R_out = kn.angle_axis_to_rotation_matrix(P[:, 0:3].view(bs, 3))
        PM = torch.cat((R_out[:,0:3,0:3], P[:, 3:6].view(bs, 3, 1)), dim=-1)
    else:
        PM = P
    pts3d_cam = pts3d_h.matmul(PM.transpose(-2,-1))
    pts2d_proj = pts3d_cam.matmul(K.t())
    S = pts2d_proj[:,:, 2].view(bs, n, 1)
    S[S==0] = S[S==0] + 1e-12
    pts2d_pro = pts2d_proj[:,:,0:2].div(S)

    return pts2d_pro






