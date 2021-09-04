###################
# import packages #
###################

import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, CosineAnnealingWarmRestarts
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion

import torch.optim as optim
import wandb
import torchvision
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import pandas as pd


#############
# Functions #
#############

#Hold random seed
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# get learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        train_acc = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in train_acc if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

# sampler를 사용할 때에는 index를 조작해야 하기 때문에 shuffle=False로 설정해야 합니다. 
def getDataloader(dataset, train_idx, valid_idx, batch_size, num_workers):
    # 인자로 전달받은 dataset에서 train_idx에 해당하는 Subset 추출
    train_set = torch.utils.data.Subset(dataset, indices=train_idx)
    # 인자로 전달받은 dataset에서 valid_idx에 해당하는 Subset 추출
    val_set   = torch.utils.data.Subset(dataset, indices=valid_idx)
    
    # 추출된 Train Subset으로 DataLoader 생성
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
    )
    # 추출된 Valid Subset으로 DataLoader 생성
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=False
    )
    
    # 생성한 DataLoader 반환
    return train_loader, val_loader

def print_confusion_matrix(cm):
    length = cm.shape[0]
    for i in range(length):
        print("%3d"%i, end='')
    print()
    print("   __"*length)
    for idx in range(length):
        print("%2d|"%idx,end='')
        print()
        for jdx in range(length):
            print("%4d"%cm[idx,jdx],end='|')
        print()
        print("   __"*length)

def rand_bbox(size, lam, method): # size : [Batch_size, Channel, Width, Height]
    """
    cut_mix function
    """
    W = size[2] 
    H = size[3] 

    if method == 'cutout':
        cut_rat = np.sqrt(1. - lam)/1.5  # 패치 크기 비율
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)  
        
        # 패치의 중앙 좌표 값 cx, cy
        cx = np.random.randint(W*0.7, W)
        cy = np.random.randint(H*0.3, H*0.7) 

    elif method == 'cutmix':  
        cut_rat = np.sqrt(1. - lam)  # 패치 크기 비율
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)  

        # 패치의 중앙 좌표 값 cx, cy
        cx = np.random.randint(W*0.4)
        cy = np.random.randint(H*0.4, H*0.8) 

    # 패치 모서리 좌표 값 
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
   
    return bbx1, bby1, bbx2, bby2


def train(data_dir, model_dir, args):

    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.wandb_name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = args.batch_size
    num_workers = 4


    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)
    dataset = dataset_module(
        label = args.label, data_dir=data_dir
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: CustomAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # 5-fold Stratified KFold 5개의 fold를 형성하고 5번 Cross Validation을 진행합니다.
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits)

    labels = [dataset.encode_multi_class(mask, gender, age) for mask, gender, age in zip(dataset.mask_labels, dataset.gender_labels, dataset.age_labels)]
    for fold, (train_idx, valid_idx) in enumerate(skf.split(dataset.image_paths, labels)):
        # -- data_loader
        # 생성한 Train, Valid Index를 getDataloader 함수에 전달해 train/valid DataLoader를 생성합니다.
        # 생성한 train, valid DataLoader로 이전과 같이 모델 학습을 진행합니다. 
        train_loader, val_loader = getDataloader(dataset, train_idx, valid_idx, batch_size, num_workers)

        # -- model
        model_module = getattr(import_module("model"), args.model)  # default: BaseModel
        model = model_module(
            num_classes=num_classes
        ).to(device)
        model = torch.nn.DataParallel(model)

        # -- loss & metric
        criterion = create_criterion(args.criterion)  # default: cross_entropy
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=0.001)

        # -- logging
        logger = SummaryWriter(log_dir=save_dir)
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)

        # -- parameter
        NUM_EPOCH = args.epochs
        BATCH_SIZE = args.batch_size
        LEARNING_RATE = args.lr
        SCHEDULAR = "CosineAnnealingWarm"
        CREITERION = args.criterion
        AUGMENTATION = args.augmentation
        VAL_SPLIT = args.val_ratio
        DATASET = args.dataset
        DATA_AUGMENT = args.data_argument
        BETA = args.beta
        PROB = args.cut_prob
        LABEL = args.label
        NAME = args.wandb_name
        
        # -- wandb
        wandb.login()
        config = {
        'epochs': NUM_EPOCH, 'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE, 'Schedular': SCHEDULAR, 'Criterion': CREITERION,
        'val_split': VAL_SPLIT,  'Augmentation': AUGMENTATION, 'Dataset': DATASET, 'cut.beta': BETA, 'cut.prob': PROB, 'Label': LABEL, 'exp': NAME
        }

        wandb.init(project='image-classification-mask', entity='team-34', config=config)
        # Fold별로 저장
        wandb.run.name = args.wandb_name + str(fold)

        wandb.watch(model)
        
        
        # -- Train
        best_val_acc = 0
        best_val_f1 = 0
        best_val_loss = np.inf

        df = pd.DataFrame()
        for epoch in range(args.epochs):
            # train loop
            model.train()
            train_loss = 0
            train_acc = 0
            train_f1 = 0

            for idx, train_batch in enumerate(train_loader):
                inputs, labels, path, state = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                r = np.random.rand(1)
       
                if args.data_argument == 'cutout' and  r < args.cut_prob:     
                    lam = np.random.beta(args.beta, args.beta)
                    target_a = labels 
                    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam, args.data_argument)
                    inputs[:, :, bbx1:bbx2, bby1:bby2] = 0
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                    outs = model(inputs)
                    loss = criterion(outs, target_a)

                elif args.data_argument == 'cutmix' and  r < args.cut_prob:
                    lam = np.random.beta(args.beta, args.beta)
                    rand_index = torch.randperm(inputs.size()[0]).to(device)
                    target_a = labels 
                    target_b = labels[rand_index]        
                    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam, args.data_argument)
                    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                    outs = model(inputs)
                    loss = criterion(outs, target_a) * lam + criterion(outs, target_b) * (1. - lam)
                
                else:
                    outs = model(inputs)
                    loss = criterion(outs, labels)      
                
                
                preds = torch.argmax(outs, dim=-1)
            
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_acc += (preds == labels).sum().item()
                train_f1 += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')

                if (idx + 1) % args.log_interval == 0:
                    middle_train_loss = train_loss / (idx + 1)
                    middle_train_acc = train_acc / (args.batch_size * (idx+1))
                    middle_f1 = train_f1/(idx+1)

                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {middle_train_loss:4.4}  || training f1 {middle_f1:4.2%} || training accuracy {middle_train_acc:4.2%} || lr {current_lr}"
                    )
            
            final_train_loss = train_loss / (len(train_loader.dataset))
            final_train_acc = train_acc / (len(train_loader.dataset))
            final_train_f1 = train_f1/(idx+1)

            scheduler.step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss = 0
                figure = None
                cm = np.zeros((num_classes,num_classes))    

                pred_list = [] 
                labels_list = []
                path_list = []
                for idx, val_batch in  enumerate(val_loader):
                    inputs, labels, path, state = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)
                    
                    val_loss += criterion(outs, labels).item()
                    
                    labels_list.extend(labels.cpu().tolist())
                    pred_list.extend(preds.cpu().tolist())
                    path_list.extend(path)

                    cm += confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(),labels=list(range(num_classes)))

                df[f"epoch_{epoch}_path"] = path_list
                df[f"epoch_{epoch}_pred"] = pred_list
                df[f"epoch_{epoch}_label"] = labels_list

                val_f1 = f1_score(labels_list, pred_list, average='macro')
                val_acc = sum((df[f"epoch_{epoch}_pred"] == df[f"epoch_{epoch}_label"]))/len(df)
                val_loss = val_loss / len(val_loader.dataset)
                best_val_loss = min(best_val_loss, val_loss)

                if epoch % 5 == 4:
                    plt.figure(figsize=(15,13))
                    plt.title("Validation CM %s"%args.wandb_name)
                    sns.heatmap(cm.astype(int), cmap='Blues', annot=True, fmt="d")
                    plt.savefig(f"{save_dir}/{fold}_{epoch}_Confusion.png")

                if val_acc > best_val_acc or val_f1 > best_val_f1:
                    print(f"New best model for val accuracy or f1 : {val_acc:4.2%}|| {val_f1:4.2%}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{save_dir}/{fold}_{epoch}_accuracy_{val_acc:4.2%}_f1_{val_f1:4.2%}.pth")
                    best_val_acc = val_acc
                    best_val_f1 = val_f1
                print(
                    f"[Val] acc : {val_acc:4.2%}, f1 : {val_f1:4.2%}, loss: {val_loss:4.2} || "
                    f"best f1 : {best_val_f1:4.2%}, best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )

                print("Print Validaition Confusion Matrix..")
                print_confusion_matrix(cm)

                # wandb 검증 단계에서 Loss, Accuracy 로그 저장
                wandb.log({
                    "train loss": final_train_loss,
                    "train acc" : final_train_acc,
                    "train f1": final_train_f1,
                    "validation loss": val_loss,
                    "validation acc" : val_acc, 
                    "validation f1": val_f1,
                })

        df.to_csv(f"{save_dir}/fold_{fold}_{args.label}.csv", index=False)
        wandb.finish()
        
        
########
# main #
########
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 30)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskSplitByProfileDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[224, 224], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='ResNet18', help='model type (default: ResNet18)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type (default: AdamW)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='focal', help='criterion type (default: focal)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--wandb_name', required=True, type=str, default='name_nth_modelname', help='model name shown in wandb. (Usage: name_nth_modelname, Example: seyoung_1st_resnet18')
    parser.add_argument('--label', required=True, type=str, default='label', help='set label : age, gender, mask, label')
    parser.add_argument('--n_split', required=False, type=int, default = 5, help='set split num')

    
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '../Input/data/train/newimages'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    # cutmix setting  
    parser.add_argument('--beta', default=0, type=float, help='hyperparameter beta')
    parser.add_argument('--cut_prob', default=0, type=float, help='cut probability')
    parser.add_argument('--data_argument', default=0, type=str, help='choose data argument. (example = cutmix, cutout)') 


    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
