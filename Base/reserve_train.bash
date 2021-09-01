#!/bin/bash

# python train.py --wandb_name "Jin_resnet_newcutmix5" --cutmix_prob 0 --epochs 15
python train_by_CLASS.py --wandb_name "Jin_resnet_custom_age_withcutmix" --beta 0.1 --cutmix_prob 0.2 --epochs 15 --label age --criterion cross_entropy
python train_by_CLASS.py --wandb_name "Jin_resnet_custom_age" --epochs 15 --label age --criterion cross_entropy


