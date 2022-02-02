from yacs.config import CfgNode as CN
from libs.process_result import Result_processor
from libs.utils import get_logger

def main(cfg):
    logger = get_logger(cfg)
    processor = Result_processor(cfg, mat_file=cfg.OUTPUT_DIR+'/'+cfg.obj+'/'+cfg.log_name+'_result.mat')
    processor.ycbv_auc(logger=logger)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--obj', required=True, type=str)
    parser.add_argument('--log_name', required=True, type=str)
    args = parser.parse_args()
    cfg = CN(new_allowed=True)
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.obj = args.obj
    cfg.log_name = args.log_name 

    main(cfg)










