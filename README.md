# image-classification-level1-34
Code for image-classification Competition by AiStage
To read the detailed report, please, refer to [team report](https://docs.google.com/document/d/11l_-q7Ty8surMZOrsSj1mW8eUQlU4gp7/edit?usp=sharing&ouid=111990686605667040701&rtpof=true&sd=true)

## Requirements
1. pytorch 1.6.0
2. torchvision 0.7.0
3. pandas 1.1.5
4. opencv-python 4.5.1.48
5. scikit-learn 0.24.1
6. matplotlib 3.2.1

## Hardware
The following specs were to create original solution.
- GPU : Tesla V100 (32GB) (1GPU for 1Server)
- CPU : 8 X vCPU
- RAM : 90G

## Dataset
### Downloab and extract train.tag.gz to Data/input directory.
```
$ wget -d https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000074/data/train.tar.gz
$ tar -zxvf train.tar.gz
```

### Crop images
```
$ python crop.py
$ python eval_crop.py
```

## Train
### Train models
To train models, run follwing commands.
```
$ python train.py SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir]
```
You can add arguments for more detailed train.

### Pretrained models
You can download pretrained model that used for my trian from [Link](https://pytorch.org/hub/pytorch_vision_resnet/)

## Inference
If trained weights are prepared, you can create submission files that contains label of images.
```
$ python inference.py
```
