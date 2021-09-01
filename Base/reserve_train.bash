#!/bin/bash
python train.py --wandb_name "Jin_resnet_newcutmix_1" --cutmix_prob 0.6
python train.py --wandb_name "Jin_resnet_newcutmix_2" --cutmix_prob 0.8