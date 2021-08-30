# TODO: train.py에 잘 녹여내야 할 듯?

import wandb

wandb.login()

# Model parameter settings
# 학습모델에 대한 저장하고 싶은 정보를 config로 생성 -> 추후 테이블에서 확인 가능
NUM_EPOCH = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2
OPTIMIZER = 'ADAM'
SCHEDULAR = 'Cosineannealinglr'
AUGMENTATION = 'filp'

# Config 값으로 테이블에서 확인 가능. (세영님 노션 기준으로 생성해봤습니다.)
config = {
    'epochs': NUM_EPOCH, 'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE,
    'val_split': VAL_SPLIT, 'Schedular': SCHEDULAR,  'Augmentation': AUGMENTATION
}

# wandb init
wandb.init(project='image-classification-mask', 
            entity='team-34', 
            config=config
            ) # project는 옆에 이름으로 생성해두었습니다.
wandb.run.name = 'seyoung_1th_resNet18' # 회차 이름 명명 규칙 필요할 것 같음. ex) {이름}_{회차}_{모델명} 

wandb.wawtch(mask_resnet18)

# 그래프로 추적할 값 세팅
wandb.log({
    f'{phase} loss': epoch_loss,
    f'{phase} acc': epoch_acc,
    f'{phase} f1': epoch_f1,
})

wandb.finish()