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
import seaborn as sns
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion

import torch.optim as optim
import wandb
import torchvision
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

import math
from torch.optim.lr_scheduler import _LRScheduler

from sklearn import metrics


#############
# functions #
#############

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


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
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def getDataloader(dataset, train_idx, valid_idx, batch_size, num_workers):
    # 인자로 전달받은 dataset에서 train_idx에 해당하는 Subset 추출
    train_set = torch.utils.data.Subset(dataset,
                                        indices=train_idx)
    # 인자로 전달받은 dataset에서 valid_idx에 해당하는 Subset 추출
    val_set   = torch.utils.data.Subset(dataset,
                                        indices=valid_idx)
    
    # 추출된 Train Subset으로 DataLoader 생성
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=True
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


def rand_bbox(size, lam): # size : [Batch_size, Channel, Width, Height]
    """

    cut_mix function

    """
    W = size[2] 
    H = size[3] 
    cut_rat = np.sqrt(1. - lam)  # 패치 크기 비율
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)  

   	# 패치의 중앙 좌표 값 cx, cy
    # cx = np.random.randint(W*0.4)
    # cy = np.random.randint(H*0.4, H*0.8) 
    

    # 패치 모서리 좌표 값 
    bbx1 = np.clip(np.int(W/2) - cut_w, 0, W)
    bby1 = np.clip(np.int(H//2) - cut_h, 0, H//2)
    bbx2 = np.clip(np.int(W/2) + cut_w, 0, W)
    bby2 = np.clip(np.int(H//2) + cut_h, 0, H//2)
   
    return bbx1, bby1, bbx2, bby2


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


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
        data_dir=data_dir,
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
        model = model_module().to(device)
        model = torch.nn.DataParallel(model)

        # -- loss & metric
        criterion = create_criterion(args.criterion)  # default: cross_entropy
        """
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        )
        """
        optimizer = optim.AdamW(model.parameters(),lr=1e-6)
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=2, eta_max=0.1,  T_up=3, gamma=0.5)

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
        BETA = args.beta
        PROB = args.cutmix_prob
        patient_max = 5
        ACCUM_size = 2
        
        # -- wandb
        wandb.login()
        config = {
        'epochs': NUM_EPOCH, 'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE, 'Schedular': SCHEDULAR, 'Criterion': CREITERION,
        'val_split': VAL_SPLIT,  'Augmentation': AUGMENTATION, 'Dataset': DATASET, 'cutmix.beta': BETA, 'cutmix.prob': PROB
        }

        wandb.init(project='image-classification-mask', 
                entity='team-34', 
                config=config
                ) 
        wandb.run.name = args.wandb_name + str(fold)

        wandb.watch(model)
        
        best_val_age_acc = 0
        best_val_age_f1 = 0
        best_val_gender_acc = 0
        best_val_gender_f1 = 0
        best_val_mask_acc = 0
        best_val_mask_f1 = 0

        best_val_loss = np.inf
        patient = 0


        for epoch in range(args.epochs):
            # train loop
            model.train()
            loss_value=0
            matches_age = 0
            matches_gender = 0
            matches_mask = 0
            train_f1_age = 0
            train_f1_gender = 0
            train_f1_mask = 0
            n_iter = 0

            for idx, train_batch in enumerate(train_loader):
                # print(len(train_batch))
                inputs, mask_lbl, gender_lbl, age_lbl = train_batch
                # mask_lbl, gender_lbl, age_lbl = dataset.decode_multi_class(batch_label)
                inputs = inputs.to(device)
                mask_lbl, gender_lbl, age_lbl = mask_lbl.to(device), gender_lbl.to(device), age_lbl.to(device)

                optimizer.zero_grad()   

                r = np.random.rand(1)
                if args.beta > 0 and  r < args.cutmix_prob:     
                    lam = np.random.beta(args.beta, args.beta)
                    rand_index = torch.randperm(inputs.size()[0]).to(device)
                    mask_cut_target, gender_cut_target, age_cut_target = dataset.decode_multi_class(labels[rand_index])        
                    
                    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                    
                    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                    
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                    
                    pred_age, pred_gender, pred_mask = model(inputs)

                    loss = criterion(pred_age, age_lbl) * lam + criterion(pred_age, age_cut_target) * (1. - lam)
                    loss += criterion(pred_gender, gender_lbl) * lam + criterion(pred_gender, gender_cut_target) * (1. - lam)   
                    loss += criterion(pred_mask, mask_lbl) * lam + criterion(pred_mask, mask_cut_target) * (1. - lam) 

                else:
                    pred_age, pred_gender, pred_mask = model(inputs)
                    loss = criterion(pred_age, age_lbl) + criterion(pred_gender, gender_lbl) + criterion(pred_mask, mask_lbl)

                # preds = torch.argmax(outs, dim=-1)
                pred_age_idx = torch.argmax(pred_age, dim=-1)
                pred_gender_idx = torch.argmax(pred_gender, dim=-1)
                pred_mask_idx = torch.argmax(pred_mask, dim=-1)


                loss.backward()

                if (idx+1)%ACCUM_size:
                    optimizer.step()
                
                train_f1_age += f1_score(age_lbl.cpu().numpy(), pred_age_idx.cpu().numpy(), average='macro')
                train_f1_gender += f1_score(gender_lbl.cpu().numpy(), pred_gender_idx.cpu().numpy(), average='macro')
                train_f1_mask += f1_score(mask_lbl.cpu().numpy(), pred_mask_idx.cpu().numpy(), average='macro')
                
                n_iter += 1

                loss_value += loss.item()

                matches_age += (pred_age_idx == age_lbl).sum().item()
                matches_gender += (pred_gender_idx == gender_lbl).sum().item()
                matches_mask += (pred_mask_idx == mask_lbl).sum().item()
                
                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc_age = matches_age/ args.batch_size / args.log_interval
                    train_acc_gender = matches_gender / args.batch_size / args.log_interval
                    train_acc_mask = matches_mask / args.batch_size / args.log_interval
                    
                    current_lr = get_lr(optimizer)
                    print(
                        f"Fold {fold}/Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4}  || lr {current_lr} ||"
                        f"train_acc_age {train_acc_age:4.2%} || train_acc_gender {train_acc_gender:4.2%} || train_acc_mask {train_acc_mask:4.2%}"
                    )
                    # logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                    # logger.add_scalar("Train/train_acc_age", train_acc_age, epoch * len(train_loader) + idx)
                    # logger.add_scalar("Train/train_acc_gender", train_acc_gender, epoch * len(train_loader) + idx)
                    # logger.add_scalar("Train/train_acc_mask", train_acc_mask, epoch * len(train_loader) + idx)

                    loss_value = 0
                    matches_age = 0
                    matches_gender = 0
                    matches_mask = 0
            
            train_f1_age /= n_iter
            train_f1_gender /= n_iter
            train_f1_mask /= n_iter
            
            wandb.log({
                        "train loss": train_loss,
                        "train_acc_age" : train_acc_age,
                        "train_acc_gender" : train_acc_gender,
                        "train_acc_mask" : train_acc_mask,
                        "train_f1_age ": train_f1_age,
                        "train_f1_gender": train_f1_gender,
                        "train_f1_mask": train_f1_mask,
                    })
            

            # 각 에폭의 마지막 input 이미지로 grid view 생성
            img_grid = torchvision.utils.make_grid(inputs)
            # Tensorboard에 train input 이미지 기록
            # logger.add_image(f'{epoch}_train_input_img', img_grid, epoch)

            # cosine annealing learing rate    
            scheduler.step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss = 0

                # figure = None

                matches_age = 0
                matches_gender = 0
                matches_mask = 0
                
                
                y_true_age = []
                y_true_gender = []
                y_true_mask = []

                y_predict_age = []
                y_predict_gender = []
                y_predict_mask = []
                val_iter = 0

                val_f1_age = 0
                val_f1_gender = 0
                val_f1_mask = 0
                
                for val_batch in val_loader:
                    val_iter += 1
                    inputs, mask_lbl, gender_lbl, age_lbl = val_batch
                    inputs = inputs.to(device)
                    mask_lbl, gender_lbl, age_lbl = mask_lbl.to(device), gender_lbl.to(device), age_lbl.to(device)


                    pred_age, pred_gender, pred_mask = model(inputs)
                    pred_age_idx = torch.argmax(pred_age, dim=-1)
                    pred_gender_idx = torch.argmax(pred_gender, dim=-1)
                    pred_mask_idx = torch.argmax(pred_mask, dim=-1)

                    loss = criterion(pred_age, age_lbl) + criterion(pred_gender, gender_lbl) + criterion(pred_mask, mask_lbl)

                    val_loss += loss.item()
                    
                    
                    matches_age += (pred_age_idx == age_lbl).sum().item()
                    matches_gender += (pred_gender_idx == gender_lbl).sum().item()
                    matches_mask += (pred_mask_idx == mask_lbl).sum().item()
                

                    # cm += confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(),labels=list(range(num_classes)))
                    y_predict_age.append(pred_age_idx.cpu().numpy())
                    y_true_age.append(age_lbl.cpu().numpy())
                    
                    y_predict_gender.append(pred_gender_idx.cpu().numpy())
                    y_true_gender.append(gender_lbl.cpu().numpy())

                    y_predict_mask.append(pred_mask_idx.cpu().numpy())
                    y_true_mask.append(mask_lbl.cpu().numpy())

                    val_f1_age += f1_score(age_lbl.cpu().numpy(), pred_age_idx.cpu().numpy(), average='macro')
                    val_f1_gender += f1_score(gender_lbl.cpu().numpy(), pred_gender_idx.cpu().numpy(), average='macro')
                    val_f1_mask += f1_score(mask_lbl.cpu().numpy(), pred_mask_idx.cpu().numpy(), average='macro')
                

                    # if figure is None:
                    #     inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    #     inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    #     figure = grid_image(
                    #         inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    #     )

                # if epoch % 5 == 4:
                #     plt.figure(figsize=(15,13))
                #     plt.title("Validation CM %s"%args.wandb_name)
                #     sns.heatmap(cm.astype(int), cmap='Blues', annot=True, fmt="d")
                #     plt.savefig(f"{save_dir}/{epoch}_Confusion.png")

                valid_acc_age = matches_age/ len(val_loader.dataset)
                valid_acc_gender = matches_gender / len(val_loader.dataset)
                valid_acc_mask = matches_mask / len(val_loader.dataset)
                    
                val_f1_age /= val_iter
                val_f1_gender /= val_iter
                val_f1_mask /= val_iter

                
                val_loss = val_loss / len(val_loader.dataset)

                best_val_loss = min(best_val_loss, val_loss)

                y_pred_age_list = [a.squeeze().tolist() for a in y_predict_age]
                y_true_age_list = [a.squeeze().tolist() for a in y_true_age]

                y_pred_gender_list = [a.squeeze().tolist() for a in y_predict_gender]
                y_true_gender_list = [a.squeeze().tolist() for a in y_true_gender]

                y_pred_mask_list = [a.squeeze().tolist() for a in y_predict_mask]
                y_true_mask_list = [a.squeeze().tolist() for a in y_true_mask]
                
                # print('===========================================================')
                # print('======================AGE============================')
                # print(metrics.classification_report(y_true_age_list, y_pred_age_list))
                # print('===========================================================')
                # print('======================GENDER============================')
                # print(metrics.classification_report(y_true_gender_list, y_pred_gender_list))
                # print('===========================================================')
                # print('======================MASK============================')
                # print(metrics.classification_report(y_true_mask_list, y_pred_mask_list))
                # print('===========================================================')
                # print('===========================================================')



                best_val_age_acc = max(best_val_age_acc, valid_acc_age)
                best_val_age_f1 = max(best_val_age_f1, val_f1_age)
                
                best_val_gender_acc =max(best_val_gender_acc, valid_acc_gender)
                best_val_gender_f1 =max(best_val_gender_f1, val_f1_gender)

                best_val_mask_acc = max(best_val_mask_acc, valid_acc_mask)
                best_val_mask_f1 = max(best_val_mask_f1, val_f1_mask)

                if val_f1_age == best_val_age_f1 and val_f1_gender == best_val_gender_f1 and val_f1_mask == best_val_mask_f1 :
                    print(f"New best model for val f1(AGE / GENDER/ MASK) : {best_val_age_f1:4.2%}|| {best_val_gender_f1:4.2%}||{best_val_mask_f1:4.2%} ! saving the best model..")
                    print(f"val_f1_age {val_f1_age:4.2%} || val_f1_gender {val_f1_gender:4.2%} || valid_acc_mask {valid_acc_mask:4.2%}")

                    torch.save(model.module.state_dict(), f"{save_dir}/{fold}_{epoch}_f1(AGE_GENDER_MASK) : {best_val_age_f1:4.2%}|| {best_val_gender_f1:4.2%}||{best_val_mask_f1:4.2%}.pth")
                    patient = 0
                else:
                    patient += 1
                
                if patient > patient_max:
                    print(f"\nValid loss didn't improve last {patient} epochs.")
                    break

        

                # print("Print Validaition Confusion Matrix..")
                # # print_confusion_matrix(cm)
                # print(metrics.classification_report())

                # logger.add_scalar("Val/loss", val_loss, epoch)
                # logger.add_scalar("Val/accuracy", val_acc, epoch)
                # logger.add_figure("results", figure, epoch)
                print()

                # wandb 검증 단계에서 Loss, Accuracy 로그 저장
                wandb.log({
                        "val_loss": val_loss,
                        "valid_acc_age" : valid_acc_age,
                        "valid_acc_gender" : valid_acc_gender,
                        "valid_acc_mask" : valid_acc_mask,
                        "val_f1_age ": val_f1_age,
                        "val_f1_gender": val_f1_gender,
                        "val_f1_mask": val_f1_mask,
                })
        print(f"fold {fold}번째 best model for val f1(AGE / GENDER/ MASK) : {best_val_age_f1:4.2%}|| {best_val_gender_f1:4.2%}||{best_val_mask_f1:4.2%}")}번째 best model for val f1(AGE / GENDER/ MASK) : {best_val_age_f1:4.2%}|| {best_val_gender_f1:4.2%}||{best_val_mask_f1:4.2%}")

        wandb.finish()

    '''
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                    accuracy))
                                                    '''
########
# main #
########

if __name__ == '__main__':
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 30)')
    parser.add_argument('--dataset', type=str, default='MaskMultiTaskingDataset', help='dataset augmentation type (default: MaskSplitByProfileDataset)')
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation', help='data augmentation type (default: CustomAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[224, 224], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='efficientnet_multitask', help='model type (default: ResNet18)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type (default: AdamW)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='focal', help='criterion type (default: focal)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--wandb_name', required=True, type=str, default='name_nth_modelname', help='model name shown in wandb. (Usage: name_nth_modelname, Example: seyoung_1st_resnet18')
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    # cutmix setting
    parser.add_argument('--beta', default=0, type=float, help='hyperparameter beta')
    parser.add_argument('--cutmix_prob', default=0.2, type=float, help='cutmix probability')

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
