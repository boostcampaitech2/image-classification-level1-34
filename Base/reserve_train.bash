#!/bin/bash
#python train_by_CLASS.py --wandb_name "Jin_resnet_cutout_gender" --beta 0.1 --cutmix_prob 0.5 --label gender --epochs 15 --criterion label_smoothing 
#python train_by_CLASS.py --wandb_name "Jin_resnet_cutout_gender" --val 0.1 --beta 0.1 --cutmix_prob 0.5 --label age --epochs 15 --criterion label_smoothing 

python train_by_CLASS.py --wandb_name "Jin_resnet_gender_mask" --beta 0.1 --cutmix_prob 0.5 --epochs 15 --criterion label_smoothing 

#python train_by_CLASS.py --wandb_name "Jin_resnet_age_withcutout_ls_(0.2)" --beta 0.1 --cutmix_prob 0.2 --label age --criterion label_smoothing 


#python train_by_CLASS.py --wandb_name "Jin_resnet_age_withcutout_ls_(0.5)" --beta 0.1 --cutmix_prob 0.5 --label age --criterion label_smoothing
#python train_by_CLASS.py --wandb_name "Jin_resnet_age_withcutout_(0.5)" --beta 0.1 --cutmix_prob 0.5 --label age 



#python train_by_CLASS.py --wandb_name "Jin_resnet_age_withcutmix5" --beta 0.1 --cutmix_prob 0.2 --epochs 15 --label age --criterion label_smoothing


#python inference.py --label gender --model_dir ./model/Jin_resnet_cutout_gender --model_path 0_13_accuracy_95.04%_f1_94.80%.pth
#python inference.py --label gender --model_dir ./model/Jin_resnet_cutout_gender --model_path 1_14_accuracy_95.66%_f1_95.45%.pth
#python inference.py --label gender --model_dir ./model/Jin_resnet_cutout_gender --model_path 2_14_accuracy_97.36%_f1_97.22%.pth
#python inference.py --label gender --model_dir ./model/Jin_resnet_cutout_gender --model_path 3_14_accuracy_94.96%_f1_94.72%.pth
#python inference.py --label gender --model_dir ./model/Jin_resnet_cutout_gender --model_path 4_13_accuracy_96.90%_f1_96.72%.pth
