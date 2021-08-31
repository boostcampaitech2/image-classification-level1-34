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
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion

import torch.optim as optim
import wandb
import torchvision
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

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
"""
cut_mix function
"""
def rand_bbox(size, lam): # size : [Batch_size, Channel, Width, Height]
    W = size[2] 
    H = size[3] 
    cut_rat = np.sqrt(1. - lam)  # 패치 크기 비율
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)  

   	# 패치의 중앙 좌표 값 cx, cy
    cx = np.random.randint(W)
    cy = np.random.randint(H)
		
    # 패치 모서리 좌표 값 
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
   
    return bbx1, bby1, bbx2, bby2


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


def train(data_dir, model_dir, args):

    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.wandb_name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = 64
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
        model = model_module(
            num_classes=num_classes
        ).to(device)
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
        optimizer = optim.AdamW(model.parameters(),lr=1e-3)
        #scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

        # -- logging
        logger = SummaryWriter(log_dir=save_dir)
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)

        # -- parameter
        NUM_EPOCH = args.epochs
        BATCH_SIZE = args.batch_size
        LEARNING_RATE = args.lr
        #SCHEDULAR = "CosineAnnealingLR"
        AUGMENTATION = args.augmentation
        VAL_SPLIT = args.val_ratio
        DATASET = args.dataset
        
        # -- wandb
        wandb.login()
        config = {
        'epochs': NUM_EPOCH, 'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE,
        'val_split': VAL_SPLIT,  'Augmentation': AUGMENTATION, 'Dataset': DATASET
        }

        wandb.init(project='image-classification-mask', 
                entity='team-34', 
                config=config
                ) 
        wandb.run.name = args.wandb_name + str(fold)

        wandb.watch(model)
        
        best_val_acc = 0
        best_val_f1 = 0
        best_val_loss = np.inf


        for epoch in range(args.epochs):
            # train loop
            model.train()
            loss_value = 0
            matches = 0
            train_f1 = 0
            val_f1 = 0
            n_iter = 0

            for idx, train_batch in enumerate(train_loader):
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()   

                r = np.random.rand(1)
                if args.beta > 0 and  r < args.cutmix_prob:     
                    lam = np.random.beta(args.beta, args.beta)
                    rand_index = torch.randperm(inputs.size()[0]).to(device)
                    target_a = labels 
                    target_b = labels[rand_index]        
                    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
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
                #scheduler.step()
                
                train_f1 += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
                n_iter += 1

                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    
                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    )
                    logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                    logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                    loss_value = 0
                    matches = 0
            
            train_f1 /= n_iter
            wandb.log({
                        "train loss": train_loss,
                        "train acc" : train_acc,
                        "train f1": train_f1,
                    })
            

            # 각 에폭의 마지막 input 이미지로 grid view 생성
            img_grid = torchvision.utils.make_grid(inputs)
            # Tensorboard에 train input 이미지 기록
            logger.add_image(f'{epoch}_train_input_img', img_grid, epoch)

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                figure = None
                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                    if figure is None:
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(
                            inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                        )
                val_f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(valid_idx) ## 이상함 ㅋㅋㅋㅋㅋ
                best_val_loss = min(best_val_loss, val_loss)
                if val_acc > best_val_acc or val_f1 > best_val_f1:
                    print(f"New best model for val accuracy or f1 : {val_acc:4.2%}|| {val_f1:4.2%}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{save_dir}/{fold}_{epoch}_accuracy_{val_acc:4.2%}_f1_{val_f1:4.2%}.pth")
                    best_val_acc = val_acc
                    best_val_f1 = val_f1
                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best f1 : {best_val_f1:4.2%},best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
                logger.add_scalar("Val/loss", val_loss, epoch)
                logger.add_scalar("Val/accuracy", val_acc, epoch)
                logger.add_figure("results", figure, epoch)
                print()

                # wandb 검증 단계에서 Loss, Accuracy 로그 저장
                wandb.log({
                    "validation loss": val_loss,
                    "validation acc" : val_acc, 
                    "validation f1": val_f1,
                })

        wandb.finish()
    '''
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                    accuracy))
                                                    '''
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskSplitByProfileDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='ResNet18', help='model type (default: ResNet18)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type (default: AdamW)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='label_smoothing', help='criterion type (default: label_smoothing)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--wandb_name', type=str, default='JIN_5th_ResMLP with cutmix, Adam', help='model name shown in wandb.')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '../Input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    # cutmix setting
    parser.add_argument('--beta', default=0.1, type=float, help='hyperparameter beta')
    parser.add_argument('--cutmix_prob', default=0, type=float, help='cutmix probability')

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)


