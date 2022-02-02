import torch
from . import utils
from scipy.io import loadmat, savemat


class Result_processor(object):
    def __init__(self, cfg, mat_file):
        temp = loadmat(mat_file)
        self.test_idx = temp['test_idx'].squeeze()
        self.cfg = cfg
        self.result_dict = temp
        self.mat_file = mat_file

    def ycbv_auc(self, logger=None):
        poses = torch.tensor(self.result_dict['pose_recordc']).float()
        pts_model_h, is_sym = utils.get_ycbv_3dmodel(self.cfg,homo=True) 
        annos = loadmat(self.cfg.YCBV_DIR+'/test_annos/obj{:02d}.mat'.format(utils.get_ycbv_objid(self.cfg.obj)))
        PMs = torch.tensor(annos['PMs']).float()
        Xs = []
        Ys = []
        for i in range(51):
            x = 0.02*i
            Xs.append(x)
            if is_sym:
                n, N = utils.ADDS_accuracy(poses, pts_model_h, PMs, 100*x, P_is_matrix=False)
            else:
                n, N = utils.ADD_accuracy(poses, pts_model_h, PMs, 100*x, P_is_matrix=False)
            assert N==len(self.test_idx)
            y = n/N
            Ys.append(y)
            logger.info('Computing ADD(-S) for AUC, x:{:.4f}, y:{:.4f}'.format(x,y))
        import sklearn.metrics as M
        auc = M.auc(Xs, Ys)
        if logger is not None:
            logger.info('Obj: {}, AUC: {:1.4f}'.format(self.cfg.obj, auc))
        self.result_dict.update({'Xs':Xs, 'Ys':Ys, 'auc':auc})
        savemat(self.mat_file, self.result_dict)





