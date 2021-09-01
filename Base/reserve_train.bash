#!/bin/bash
python train.py --wandb_name "Jin_resnet_newcutmix_2" --cutmix_prob 0.8
python train.py --wandb_name "Jin_resnet_newcutmix_3" --cutmix_prob 0.4