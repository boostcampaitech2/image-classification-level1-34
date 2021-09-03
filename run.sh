#!/bin/bash

/opt/conda/bin/python /opt/ml/image-classification-level1-34/Base/train_by_label.py --label "age" --wandb_name "age_resnext50_labelsmoot" --epoch 8 --model "Resnext50" --fold 5

