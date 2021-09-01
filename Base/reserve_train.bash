#!/bin/bash

# python train.py --wandb_name "Jin_resnet_newcutmix5" --cutmix_prob 0 --epochs 15
python train.py --wandb_name "Jin_resnet_newcutmix6" --beta 0.1 --cutmix_prob 0.3 --epochs 15