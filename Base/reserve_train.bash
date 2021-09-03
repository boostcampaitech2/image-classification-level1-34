#!/bin/bash

# python train.py --wandb_name "Jin_resnet_newcutmix5" --cutmix_prob 0 --epochs 15

python train_by_CLASS.py --wandb_name "Jin_resnet_age_withcutout" --beta 0.1 --cutmix_prob 0.5 --epochs 10 --label age 

#python train_by_CLASS.py --wandb_name "Jin_resnet_age_withcutmix5" --beta 0.1 --cutmix_prob 0.2 --epochs 15 --label age --criterion label_smoothing

