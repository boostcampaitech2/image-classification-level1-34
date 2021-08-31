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
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data.sampler import WeightedRandomSampler

from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion

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
        train_acc = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in train_acc if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

# sampler를 사용할 때에는 index를 조작해야 하기 때문에 shuffle=False로 설정해야 합니다. 
def getDataloader(dataset, train_idx, valid_idx, batch_size, num_workers, sampler=None):
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
        sampler=sampler,
        shuffle=False
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

def make_weights_for_balanced_classes(labels, nclasses=18):                        
    count = [0] * nclasses
    #라벨 개수를 count에 저장
    for label in labels:
        count[label] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(labels)                                              
    for idx, val in enumerate(labels):                                          
        weight[idx] = weight_per_class[val]                                  
    return weight 

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
        if args.sampler==True:
            '''
            샘플러 정의
            weighted_sampler = 어쩌고 저쩌고
            '''
            weights = make_weights_for_balanced_classes(labels, 18)
            weights = torch.DoubleTensor(weights)
            weighted_sampler = WeightedRandomSampler(weights, len(weights))
            train_loader, val_loader = getDataloader(dataset, train_idx, valid_idx, batch_size, num_workers, weighted_sampler)
        else:
            train_loader, val_loader = getDataloader(dataset, train_idx, valid_idx, batch_size, num_workers)



        # -- model
        model_module = getattr(import_module("model"), args.model)  # default: BaseModel
        model = model_module(
            num_classes=num_classes
        ).to(device)
        model = torch.nn.DataParallel(model)

        # -- loss & metric
        criterion = create_criterion(args.criterion)  # default: cross_entropy
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: AdamW
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        )
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
        CRITERION = args.criterion
        
        # -- wandb
        wandb.login()
        config = {
        'epochs': NUM_EPOCH, 'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE,
        'val_split': VAL_SPLIT,  'Augmentation': AUGMENTATION, 'Dataset': DATASET, 'Criterion': CRITERION
        # Wandb에 남기고 싶은 config log가 있다면 여기에 넣어주시면 됩니다 :)
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
            train_loss = 0
            train_acc = 0
            train_f1 = 0

            for idx, train_batch in enumerate(train_loader):
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)

                loss.backward()
                optimizer.step()
                #scheduler.step()

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
                    logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                    logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
            
            final_train_loss = train_loss / (len(train_loader))
            final_train_acc = train_acc / (len(train_loader))
            final_train_f1 = train_f1/(idx+1)

            wandb.log({
                        "train loss": final_train_loss,
                        "train acc" : final_train_acc,
                        "train f1": final_train_f1,
                    })
            

            # 각 에폭의 마지막 input 이미지로 grid view 생성
            img_grid = torchvision.utils.make_grid(inputs)
            # Tensorboard에 train input 이미지 기록
            logger.add_image(f'{epoch}_train_input_img', img_grid, epoch)
            

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                # Threshold - 이 값 넘겨야 저장
                val_acc_c = 0.6
                val_f1_c = 0.6
                val_loss = 0
                val_acc = 0
                val_f1 = 0
                figure = None
                n_iter = 0
                for idx, val_batch in enumerate(val_loader):
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    #print(len(labels))

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    
                    val_loss += loss_item
                    val_acc += acc_item
                    val_f1 += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')

                    if figure is None:
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(
                            inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                        )
                    n_iter += 1

                val_f1 = val_f1 / n_iter
                val_loss = val_loss / len(val_loader.dataset)
                val_acc = val_acc / len(val_loader.dataset)
                #print(len(val_loader))

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
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 30)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskSplitByProfileDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument('--sampler', type=bool, default=False, help='use weighted sampler (default: false)')
    parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='ResNet18', help='model type (default: ResNet18)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type (default: AdamW)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy, menu: focal, f1, label_smoothing)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--wandb_name', required=True, type=str, default='name_nth_modelname', help='model name shown in wandb. (Usage: name_nth_modelname, Example: seyoung_1st_resnet18')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
