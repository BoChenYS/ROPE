import datetime
import os
import time
from yacs.config import CfgNode as CN
from scipy.io import savemat

import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

from reference.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from reference.engine import train_one_epoch, evaluate
from reference import utils
from dataset.lm import lm_with_synt
from dataset.lmo import lmo
from detection.keypoint_rcnn import keypointrcnn_hrnet
from libs.utils import get_logger

def main(args, cfg):
    utils.init_distributed_mode(args)
    logger = get_logger(cfg)
    device = torch.device(cfg.DEVICE)

    # Data loading code
    print("Loading data")

    dataset = lm_with_synt(cfg)
    dataset_test_full = lmo(cfg)
    valid_list = dataset_test_full.img_list
    dataset_test = torch.utils.data.Subset(dataset_test_full, valid_list)

    print("Creating data loaders. Is distributed? ", args.distributed)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, cfg.BATCH_SIZE)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, cfg.BATCH_SIZE, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=cfg.WORKERS,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=cfg.TEST_BATCH_SIZE,
        sampler=test_sampler, num_workers=cfg.WORKERS,
        collate_fn=utils.collate_fn)

    print("Creating model")
    model = keypointrcnn_hrnet(cfg, resume=args.resume, min_size=480, max_size=640)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg.LR)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.LR_STEPS, gamma=cfg.LR_DECAY)

    if args.resume:
        checkpoint = torch.load(os.path.join(cfg.OUTPUT_DIR,cfg.obj,'{}.pth'.format(cfg.log_name)), map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluator = evaluate(model, data_loader_test, device=device, logger=logger)

        boxes, pose_record1, pose_record2, pose_record3, pose_recordc, pts2d_record1, pts2d_record2, pts2d_record3, \
        corrects1, corrects2, corrects3, correctsc \
        = evaluator.get_accuracy(cfg, args.start_epoch-1, n_test=len(valid_list), testset_name='lmo', n_min=4, thres=1, logger=logger)
        savemat(os.path.join(cfg.OUTPUT_DIR, cfg.obj, '{}_result.mat'.format(cfg.log_name)), 
            {'boxes':boxes.cpu().numpy(), 'pose_record1': pose_record1.detach().cpu().numpy(), 
            'pose_record2': pose_record2.detach().cpu().numpy(), 'pose_record3': pose_record3.detach().cpu().numpy(), 
            'pose_recordc': pose_recordc.detach().cpu().numpy(), 'pts2d_record1': pts2d_record1.detach().cpu().numpy(),
            'pts2d_record2': pts2d_record2.detach().cpu().numpy(), 'pts2d_record3': pts2d_record3.detach().cpu().numpy(), 
            'corrects1':corrects1.detach().cpu().numpy(), 'corrects2':corrects2.detach().cpu().numpy(), 
            'corrects3':corrects3.detach().cpu().numpy(), 'correctsc':correctsc.detach().cpu().numpy(), 'test_idx': valid_list})
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, cfg.END_EPOCH):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_one_epoch(model, optimizer, data_loader, device, epoch, cfg.PRINT_FREQ, cfg.obj, logger)
        lr_scheduler.step()
        if cfg.OUTPUT_DIR:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'cfg': cfg,
                'epoch': epoch},
                os.path.join(cfg.OUTPUT_DIR, cfg.obj, '{}.pth'.format(cfg.log_name)))

        if epoch==cfg.END_EPOCH-1:
            evaluator = evaluate(model, data_loader_test, device=device, logger=logger)
            boxes, pose_record1, pose_record2, pose_record3, pose_recordc, pts2d_record1, pts2d_record2, pts2d_record3, \
            corrects1, corrects2, corrects3, correctsc \
            = evaluator.get_accuracy(cfg, epoch, n_test=len(valid_list), testset_name='lmo', n_min=4, thres=1, logger=logger)
            savemat(os.path.join(cfg.OUTPUT_DIR, cfg.obj, '{}_result.mat'.format(cfg.log_name)), 
                {'boxes':boxes.cpu().numpy(), 'pose_record1': pose_record1.detach().cpu().numpy(), 
                'pose_record2': pose_record2.detach().cpu().numpy(), 'pose_record3': pose_record3.detach().cpu().numpy(), 
                'pose_recordc': pose_recordc.detach().cpu().numpy(), 'pts2d_record1': pts2d_record1.detach().cpu().numpy(),
                'pts2d_record2': pts2d_record2.detach().cpu().numpy(), 'pts2d_record3': pts2d_record3.detach().cpu().numpy(), 
                'corrects1':corrects1.detach().cpu().numpy(), 'corrects2':corrects2.detach().cpu().numpy(), 
                'corrects3':corrects3.detach().cpu().numpy(), 'correctsc':correctsc.detach().cpu().numpy(), 'test_idx': valid_list})

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--resume', dest="resume",action="store_true")
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=-1, type=int)
    parser.add_argument("--test-only",dest="test_only",help="Only test the model",action="store_true",)
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--obj', required=True, type=str)
    parser.add_argument('--sigma1', default=1.5, required=False, type=float)
    parser.add_argument('--sigma2', default=3, required=False, type=float)
    parser.add_argument('--sigma3', default=8, required=False, type=float)
    parser.add_argument('--log_name', required=True, type=str)
    parser.add_argument('--distrib', default=1, type=int)
    args = parser.parse_args()
    cfg = CN(new_allowed=True)
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.obj = args.obj
    cfg.log_name = args.log_name
    cfg.sigma1 = args.sigma1
    cfg.sigma2 = args.sigma2
    cfg.sigma3 = args.sigma3
    cfg.freeze()

    main(args, cfg)





