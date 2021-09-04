###################
# import packages #
###################

import argparse
import os
from importlib import import_module
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import TestDataset, MaskBaseDataset

############
# Function #
############

# 모델 로드
def load_model(saved_model_dir, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )
    
    model_path = os.path.join(saved_model_dir, '4_12_accuracy_91.10%_f1_79.40%.pth')   # 필요한 모델의 path로 바꿔가며 사용
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
        Model inference
            - data_dir : str, data directory path
            - model_dir : str, model directory path
            - output_dir : str, inference result directory path
            - args : arguements that apply to model
    """
    
    #Set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    #Set the number of class
    if args.label == "age":
        num_classes = 3
    elif args.label == "gender":
        num_classes = 2
    elif args.label == "state":
        num_classes = 3
    else:
        num_classes = 18
    
    #load model
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()
    
    #set path and get information
    img_root = os.path.join(data_dir, 'newimages')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)
    
    #get each images path 
    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    
    #set inference DataLoader
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        #num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )
    
    #inference
    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())
    info['ans'] = preds
    
    #Save result
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')

    
########
# main #
########

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(224, 224), help='resize size for image when you trained (default: (224, 224))')
    parser.add_argument('--model', type=str, default='ResNet18', help='model type (default: ResNet18)')                                # need to change
    parser.add_argument('--label', required=True, type=str, default='label', help='set label : age, gender, state, label')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '../Input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/Jin_resnet_age_withcutmix72'))                           # need to change
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
    
