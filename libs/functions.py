import torch
from reference.utils import all_gather
from . import utils
import numpy as np
import cv2 as cv
import kornia as kn
import rowan
from tqdm import tqdm
from scipy.io import loadmat
import json

class Evaluator(object):
    def __init__(self):
        self.img_ids = []
        self.outputs = []
        self.img_ids_all = []
        self.outputs_all = []
        self.has_gathered = False

    def update(self, res):
        for img_id, output in res.items():
            self.img_ids.append(img_id)
            self.outputs.append(output)

    def gather_all(self):
        ids_list = all_gather(self.img_ids)
        outputs_list = all_gather(self.outputs)
        assert len(ids_list)==len(outputs_list)
        for ids, outs in zip(ids_list, outputs_list):
            self.img_ids_all.extend(ids)
            self.outputs_all.extend(outs)
        self.has_gathered = True

    def get_accuracy(self, cfg, epoch, n_test, testset_name, n_min=4, thres=1, logger=None):
        assert self.has_gathered
        n_correct_sum1 = 0
        n_correct_sum2 = 0
        n_correct_sum3 = 0
        n_correct_sumc = 0
        n_correct_sum1s = 0
        n_correct_sum2s = 0
        n_correct_sum3s = 0
        n_correct_sumcs = 0
        size_vec = torch.tensor([cfg.MODEL.IMAGE_SIZE],dtype=torch.float)
        if testset_name=='lm':
            pts3d_fps = utils.get_lm_o_fps(cfg)
            pts_model_h = utils.get_lm_o_3dmodel(cfg,homo=True) 
            n_dataset = utils.get_lm_dataset_size(cfg)
        if testset_name=='lmo':
            pts3d_fps = utils.get_lm_o_fps(cfg)
            pts_model_h = utils.get_lm_o_3dmodel(cfg,homo=True)
            n_dataset = 1214
        pts3d_h = torch.cat((pts3d_fps, torch.ones(cfg.N_PTS, 1)), dim=-1)
        K = utils.get_K()
        diameter = utils.get_distance(cfg)

        pts2d_record1 = torch.zeros(n_dataset, cfg.N_PTS, 2)
        pts2d_record2 = torch.zeros(n_dataset, cfg.N_PTS, 2)
        pts2d_record3 = torch.zeros(n_dataset, cfg.N_PTS, 2)
        boxes = torch.zeros(n_dataset, 4)
        pose_record1 = torch.zeros(n_dataset, 6)
        pose_record2 = torch.zeros(n_dataset, 6)
        pose_record3 = torch.zeros(n_dataset, 6)
        pose_recordc = torch.zeros(n_dataset, 6)
        corrects1 = torch.zeros(n_dataset, 1)
        corrects2 = torch.zeros(n_dataset, 1)
        corrects3 = torch.zeros(n_dataset, 1)
        correctsc = torch.zeros(n_dataset, 1)
        for idx, out_dict in tqdm(zip(self.img_ids_all, self.outputs_all)):
            if len(out_dict['scores'])==0:
                continue
            PM_gt = utils.get_PM_gt(cfg, testset_name, idx) 
            top1id = out_dict['scores'].argmax()
            box_pred = out_dict['boxes'][top1id]
            boxes[idx, :] = box_pred.detach()

            pts2d_out_coord1 = out_dict['keypoints1'][top1id,:,:2].unsqueeze(0)
            pts2d_out_coord2 = out_dict['keypoints2'][top1id,:,:2].unsqueeze(0)
            pts2d_out_coord3 = out_dict['keypoints3'][top1id,:,:2].unsqueeze(0)

            P1 = pnp(pts2d_out_coord1, pts3d_fps, K)
            pose_record1[idx, :] = P1.detach()
            pts2d_record1[idx, :, :] = pts2d_out_coord1.detach()
            n_correct1, _, ticks1 = utils.ADD_accuracy_withID(P1, pts_model_h, PM_gt, diameter)     
            n_correct_sum1 += n_correct1
            if cfg.obj == 'eggbox' or cfg.obj == 'glue':
                n_correct1s, _, ticks1s = utils.ADDS_accuracy_withID(P1, pts_model_h, PM_gt, diameter)
                n_correct_sum1s += n_correct1s
            corrects1[idx, :] = ticks1.detach()

            P2 = pnp(pts2d_out_coord2, pts3d_fps, K)
            pose_record2[idx, :] = P2.detach()
            pts2d_record2[idx, :, :] = pts2d_out_coord2.detach()
            n_correct2, _, ticks2 = utils.ADD_accuracy_withID(P2, pts_model_h, PM_gt, diameter)            
            n_correct_sum2 += n_correct2
            if cfg.obj == 'eggbox' or cfg.obj == 'glue':
                n_correct2s, _, ticks2s = utils.ADDS_accuracy_withID(P2, pts_model_h, PM_gt, diameter)
                n_correct_sum2s += n_correct2s
            corrects2[idx, :] = ticks2.detach()

            P3 = pnp(pts2d_out_coord3, pts3d_fps, K)
            pose_record3[idx, :] = P3.detach()
            pts2d_record3[idx, :, :] = pts2d_out_coord3.detach()
            n_correct3, _, ticks3 = utils.ADD_accuracy_withID(P3, pts_model_h, PM_gt, diameter)            
            n_correct_sum3 += n_correct3
            if cfg.obj == 'eggbox' or cfg.obj == 'glue':
                n_correct3s, _, ticks3s = utils.ADDS_accuracy_withID(P3, pts_model_h, PM_gt, diameter)
                n_correct_sum3s += n_correct3s
            corrects3[idx, :] = ticks3.detach()

            pts2d_out_coordc, consist_ids = utils.get_kp_consensus_aslist(pts2d_out_coord1, pts2d_out_coord2, pts2d_out_coord3, thres, n_min)
            Pc = [pnp(pts2d_out_coordc[i].unsqueeze(0), pts3d_fps[consist_ids[i]], K) for i in range(len(pts2d_out_coordc))]
            Pc = torch.cat(tuple(Pc), dim=0)

            pose_recordc[idx, :] = Pc.detach()
            n_correctc, _, ticksc = utils.ADD_accuracy_withID(Pc, pts_model_h, PM_gt, diameter)            
            n_correct_sumc += n_correctc
            if cfg.obj == 'eggbox' or cfg.obj == 'glue':
                n_correctcs, _, tickscs = utils.ADDS_accuracy_withID(Pc, pts_model_h, PM_gt, diameter)
                n_correct_sumcs += n_correctcs
            correctsc[idx, :] = ticksc.detach()

        if logger is not None:
            if cfg.obj == 'eggbox' or cfg.obj == 'glue':
                logger.info('Epoch {:3d} {:s} {} test, {}, s1:{}, s2:{}, s3:{}, n_min: {:d}, thres: {}, ADD1: {:1.4f}, ADD2: {:1.4f}, ADD3: {:1.4f}, \
                    ADDc: {:1.4f}, {:s}, ADD1s: {:1.4f}, ADD2s: {:1.4f}, ADD3s: {:1.4f}, ADDcs: {:1.4f}'.format(epoch, testset_name, cfg.obj, \
                        cfg.log_name, cfg.sigma1, cfg.sigma2, cfg.sigma3, n_min, thres, n_correct_sum1/n_test, n_correct_sum2/n_test, n_correct_sum3/n_test, \
                        n_correct_sumc/n_test, cfg.obj, n_correct_sum1s/n_test, n_correct_sum2s/n_test, n_correct_sum3s/n_test, n_correct_sumcs/n_test))
            else:
                logger.info('Epoch {:3d} {:s} {} test, {}, s1:{}, s2:{}, s3:{}, n_min: {:d}, thres: {}, ADD1: {:1.4f}, ADD2: {:1.4f}, ADD3: {:1.4f} \
                    ADDc: {:1.4f}'.format(epoch, testset_name, cfg.obj, cfg.log_name, cfg.sigma1, cfg.sigma2, cfg.sigma3, n_min, thres, \
                        n_correct_sum1/n_test, n_correct_sum2/n_test, n_correct_sum3/n_test, n_correct_sumc/n_test))
        return boxes, pose_record1, pose_record2, pose_record3, pose_recordc, pts2d_record1, pts2d_record2, pts2d_record3, corrects1, corrects2, \
        corrects3, correctsc


    def get_ycbv_accuracy(self, cfg, epoch, n_test, testset_name, n_min=4, thres=1, logger=None):
        assert self.has_gathered
        n_correct_sum1 = 0
        n_correct_sum2 = 0
        n_correct_sum3 = 0
        n_correct_sumc = 0
        n_correct_sum1s = 0
        n_correct_sum2s = 0
        n_correct_sum3s = 0
        n_correct_sumcs = 0
        size_vec = torch.tensor([cfg.MODEL.IMAGE_SIZE],dtype=torch.float)

        assert testset_name=='ycbv'
        pts3d_fps = utils.get_ycbv_fps(cfg)
        pts_model_h, is_sym = utils.get_ycbv_3dmodel(cfg,homo=True) 

        annos = loadmat(cfg.YCBV_DIR+'/test_annos/obj{:02d}.mat'.format(utils.get_ycbv_objid(cfg.obj)))
        seq_ids = annos['seq_ids'].squeeze()
        img_ids = annos['img_ids'].squeeze()
        Ks = annos['Ks']
        PMs = annos['PMs']
        n_dataset = len(img_ids)

        pts3d_h = torch.cat((pts3d_fps, torch.ones(cfg.N_PTS, 1)), dim=-1)
        diameter = utils.get_ycbv_distance(cfg)

        pts2d_record1 = torch.zeros(n_dataset, cfg.N_PTS, 2)
        pts2d_record2 = torch.zeros(n_dataset, cfg.N_PTS, 2)
        pts2d_record3 = torch.zeros(n_dataset, cfg.N_PTS, 2)
        # distance_record = torch.zeros(n_dataset, 1)
        boxes = torch.zeros(n_dataset, 4)
        pose_record1 = torch.zeros(n_dataset, 6)
        pose_record2 = torch.zeros(n_dataset, 6)
        pose_record3 = torch.zeros(n_dataset, 6)
        pose_recordc = torch.zeros(n_dataset, 6)
        corrects1 = torch.zeros(n_dataset, 1)
        corrects2 = torch.zeros(n_dataset, 1)
        corrects3 = torch.zeros(n_dataset, 1)
        correctsc = torch.zeros(n_dataset, 1)
        for idx, out_dict in tqdm(zip(self.img_ids_all, self.outputs_all)):
            if len(out_dict['scores'])==0:
                continue
            PM_gt = torch.tensor(PMs[idx]).float().unsqueeze(0)
            K = torch.tensor(Ks[idx]).float()
            top1id = out_dict['scores'].argmax()
            box_pred = out_dict['boxes'][top1id]
            boxes[idx, :] = box_pred.detach()

            pts2d_out_coord1 = out_dict['keypoints1'][top1id,:,:2].unsqueeze(0)
            pts2d_out_coord2 = out_dict['keypoints2'][top1id,:,:2].unsqueeze(0)
            pts2d_out_coord3 = out_dict['keypoints3'][top1id,:,:2].unsqueeze(0)

            P1 = pnp(pts2d_out_coord1, pts3d_fps, K)
            pose_record1[idx, :] = P1.detach()
            pts2d_record1[idx, :, :] = pts2d_out_coord1.detach()
            n_correct1, _, ticks1 = utils.ADD_accuracy_withID(P1, pts_model_h, PM_gt, diameter)     
            n_correct_sum1 += n_correct1
            if is_sym:
                n_correct1s, _, ticks1s = utils.ADDS_accuracy_withID(P1, pts_model_h, PM_gt, diameter)
                n_correct_sum1s += n_correct1s
            corrects1[idx, :] = ticks1.detach()

            P2 = pnp(pts2d_out_coord2, pts3d_fps, K)
            pose_record2[idx, :] = P2.detach()
            pts2d_record2[idx, :, :] = pts2d_out_coord2.detach()
            n_correct2, _, ticks2 = utils.ADD_accuracy_withID(P2, pts_model_h, PM_gt, diameter)            
            n_correct_sum2 += n_correct2
            if is_sym:
                n_correct2s, _, ticks2s = utils.ADDS_accuracy_withID(P2, pts_model_h, PM_gt, diameter)
                n_correct_sum2s += n_correct2s
            corrects2[idx, :] = ticks2.detach()

            P3 = pnp(pts2d_out_coord3, pts3d_fps, K)
            pose_record3[idx, :] = P3.detach()
            pts2d_record3[idx, :, :] = pts2d_out_coord3.detach()
            n_correct3, _, ticks3 = utils.ADD_accuracy_withID(P3, pts_model_h, PM_gt, diameter)            
            n_correct_sum3 += n_correct3
            if is_sym:
                n_correct3s, _, ticks3s = utils.ADDS_accuracy_withID(P3, pts_model_h, PM_gt, diameter)
                n_correct_sum3s += n_correct3s
            corrects3[idx, :] = ticks3.detach()

            pts2d_out_coordc, consist_ids = utils.get_kp_consensus_aslist(pts2d_out_coord1, pts2d_out_coord2, pts2d_out_coord3, thres, n_min)
            Pc = [pnp(pts2d_out_coordc[i].unsqueeze(0), pts3d_fps[consist_ids[i]], K) for i in range(len(pts2d_out_coordc))]
            Pc = torch.cat(tuple(Pc), dim=0)

            pose_recordc[idx, :] = Pc.detach()
            n_correctc, _, ticksc = utils.ADD_accuracy_withID(Pc, pts_model_h, PM_gt, diameter)            
            n_correct_sumc += n_correctc
            if is_sym:
                n_correctcs, _, tickscs = utils.ADDS_accuracy_withID(Pc, pts_model_h, PM_gt, diameter)
                n_correct_sumcs += n_correctcs
            correctsc[idx, :] = ticksc.detach()

        if logger is not None:
            if is_sym:
                logger.info('Epoch {:3d} {:s} obj{} test, {}, s1:{}, s2:{}, s3:{}, n_min: {:d}, thres: {}, ADD1: {:1.4f}, ADD2: {:1.4f}, ADD3: {:1.4f}, \
                    ADDc: {:1.4f}, {:s}, ADD1s: {:1.4f}, ADD2s: {:1.4f}, ADD3s: {:1.4f}, ADDcs: {:1.4f}'.format(epoch, testset_name, cfg.obj, \
                        cfg.log_name, cfg.sigma1, cfg.sigma2, cfg.sigma3, n_min, thres, n_correct_sum1/n_test, n_correct_sum2/n_test, n_correct_sum3/n_test, \
                        n_correct_sumc/n_test, cfg.obj, n_correct_sum1s/n_test, n_correct_sum2s/n_test, n_correct_sum3s/n_test, n_correct_sumcs/n_test))
            else:
                logger.info('Epoch {:3d} {:s} obj{} test, {}, s1:{}, s2:{}, s3:{}, n_min: {:d}, thres: {}, ADD1: {:1.4f}, ADD2: {:1.4f}, ADD3: {:1.4f} \
                    ADDc: {:1.4f}'.format(epoch, testset_name, cfg.obj, cfg.log_name, cfg.sigma1, cfg.sigma2, cfg.sigma3, n_min, thres, \
                        n_correct_sum1/n_test, n_correct_sum2/n_test, n_correct_sum3/n_test, n_correct_sumc/n_test))
        return boxes, pose_record1, pose_record2, pose_record3, pose_recordc, pts2d_record1, pts2d_record2, pts2d_record3, corrects1, corrects2, \
        corrects3, correctsc, seq_ids, img_ids



def pnp(pts2d, pts3d, K):
    bs = pts2d.size(0)
    n = pts2d.size(1)
    device = pts2d.device
    pts3d_np = np.array(pts3d.detach().cpu())
    K_np = np.array(K.cpu())
    P_6d = torch.zeros(bs,6,device=device)
    R_inv = torch.tensor([[-1,0,0],[0,-1,0],[0,0,-1]],device=device,dtype=torch.float)

    for i in range(bs):
        pts2d_i_np = np.ascontiguousarray(pts2d[i].detach().cpu()).reshape((n,1,2))
        _, rvec, T, _ = cv.solvePnPRansac(objectPoints=pts3d_np, imagePoints=pts2d_i_np, cameraMatrix=K_np, distCoeffs=None, flags=cv.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True)
        angle_axis = torch.tensor(rvec,device=device,dtype=torch.float).view(1, 3)
        T = torch.tensor(T,device=device,dtype=torch.float).view(1, 3)
        if T[0,2] < 0:
            RR = kn.angle_axis_to_rotation_matrix(angle_axis)
            RR = R_inv.matmul(RR)
            RR = rowan.from_matrix(RR.cpu(), require_orthogonal=False)
            ax = rowan.to_axis_angle(RR)
            angle_axis = torch.tensor(ax[0]*ax[1],device=device,dtype=torch.float).view(1,3)
            T = R_inv.matmul(T.t()).t()
        P_6d[i,:] = torch.cat((angle_axis,T),dim=-1)
    return P_6d














