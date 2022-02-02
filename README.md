# Robust Object Pose Estimation (ROPE)
This repo stores code used in the paper
## [Occlusion-Robust Object Pose Estimation with Holistic Representation](https://arxiv.org/pdf/2110.11636.pdf)

### Environment


### Datasets


### Usage examples
To train for the lm test set in distrubted mode
````bash
python -m torch.distributed.launch --nproc_per_node=<num_gpus_to_use> --use_env main_lm.py --cfg cfg.yaml --obj duck --log_name <name_this_experiment>
````

To train for the lmo test set in single GPU mode
````bash
CUDA_VISIBLE_DEVICES=<which_gpu> python main_lmo.py --cfg cfg.yaml --obj ape --log_name <name_this_experiment> 
````

To load trained model and test on the lmo dataset
````bash
CUDA_VISIBLE_DEVICES=<which_gpu> python main_lmo.py --cfg cfg.yaml --obj ape --log_name <which_experiment_to_load> --resume --test-only 
````


To train for the ycbv test set 
````bash
python -m torch.distributed.launch --nproc_per_node=<num_gpus_to_use> --use_env main_ycbv.py --cfg cfg.yaml --obj 01 --log_name <name_this_experiment>
````

To compute AUC for a ycbv test result
````bash
python analysis.py --cfg cfg.yaml --log_name <which_experiment_to_load> --obj 20
````


