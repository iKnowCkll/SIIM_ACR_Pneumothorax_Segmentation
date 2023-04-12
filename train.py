import ast
import numpy as np
import pandas as pd
import os
import wandb
from IPython.core.display import display
import time
import gc
from sklearn.model_selection import StratifiedKFold
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from my_dataset import My_Dataset
import warnings
from colorama import Fore, Style
from configparser import ConfigParser
from transform import DataTransforms
from utils.data_utils import set_seed
from utils.model_utils import build_model
from utils.train_utils import train_one_epoch, valid_one_epoch, fetch_scheduler

debug = False
warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
config_parser = ConfigParser()
config_parser.read('/home/chenk/model_train/kaggle/SIIM_ACR_Pneumothorax_Segmentation/mysql.cfg')
cfg = config_parser['default']
set_seed()
c_ = Fore.GREEN
sr_ = Style.RESET_ALL
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if not os.path.exists(cfg['save_path']):
    os.makedirs(cfg['save_path'])

# =========================data_path=================================
data_path = '/home/chenk/model_train/kaggle/SIIM_ACR_Pneumothorax_Segmentation/train/siimpng/train_png'
df_path = '/home/chenk/model_train/kaggle/SIIM_ACR_Pneumothorax_Segmentation/' \
          'siim_acr_pneumothorax_segmentation/train-rle.csv'
df_all = pd.read_csv(df_path)
df = df_all.drop_duplicates('ImageId')
df.loc[df[" EncodedPixels"] != "-1", 'has_mask'] = 1
df.loc[df[" EncodedPixels"] == "-1", 'has_mask'] = 0
df_with_mask = df[df[" EncodedPixels"] != "-1"]
df_without_mask = df[df[" EncodedPixels"] == "-1"]
df_without_mask_sampled = df_without_mask.sample(len(df_with_mask) + 1200, random_state=101)
# df_without_mask_sampled = df_without_mask.sample(int(len(df_with_mask)*float(cfg['sample_ratio'])), random_state=101)
df = pd.concat([df_with_mask, df_without_mask_sampled]).reset_index(drop=True)
skf = StratifiedKFold(n_splits=int(cfg['n_fold']), shuffle=True, random_state=int(cfg['seed']))
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['has_mask'])):
    # 表示对val_idx行，fold列赋值
    df.loc[val_idx, 'fold'] = fold
display(df.groupby(['fold', 'has_mask'])['ImageId'].count())

# =========================data_fold=================================
data_transforms = DataTransforms(img_size=ast.literal_eval(cfg['img_size']))
train_transform = data_transforms("train")
valid_transform = data_transforms("valid")


def prepare_loaders():
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)
    if debug:
        train_df = train_df.head(32 * 5).query("has_mask==1").reset_index(drop=True)
        valid_df = valid_df.head(32 * 3).query("has_mask==1").reset_index(drop=True)
    train_dataset = My_Dataset(train_df, data_path=data_path, transform=train_transform)
    valid_dataset = My_Dataset(valid_df, data_path=data_path, transform=valid_transform)

    train_fold_loader = DataLoader(train_dataset, batch_size=int(cfg['train_bs']) if not debug else 20,
                                   num_workers=4, shuffle=True, pin_memory=True, drop_last=False)
    valid_fold_loader = DataLoader(valid_dataset, batch_size=int(cfg['valid_bs']) if not debug else 20,
                                   num_workers=4, shuffle=False, pin_memory=True)

    return train_fold_loader, valid_fold_loader


# ==========================train_function=====================================
def run_training(net, optimizers, schdulers, num_epochs):
    wandb.watch(net, log_freq=100)

    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_dice = -np.inf
    best_jaccard = 0
    for epoch in range(1, num_epochs + 1):
        gc.collect()
        print(f'Epoch {epoch}/{num_epochs}', end='')
        train_loss = train_one_epoch(net, optimizers, schdulers,
                                     dataloader=train_loader,
                                     device=device, n_accumulate=int(cfg['n_accumulate']), epoch=epoch)

        val_loss, val_scores = valid_one_epoch(net, valid_loader,
                                               device=device, optimizer=optimizers, epoch=epoch)
        val_dice, val_jaccard = val_scores

        wandb.log({"Train Loss": train_loss,
                   "Valid Loss": val_loss,
                   "Valid Dice": val_dice,
                   "Valid Jaccard": val_jaccard,
                   "LR": schdulers.get_last_lr()[0]})
        print(f'Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}')

        # deep copy the model
        if val_dice >= best_dice:
            print(f"{c_}Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
            best_dice = val_dice
            best_jaccard = val_jaccard
            best_epoch = epoch
            run.summary["Best Dice"] = best_dice
            run.summary["Best Jaccard"] = best_jaccard
            run.summary["Best Epoch"] = best_epoch
            save_path = cfg['save_path'] + f"best_epoch-{fold:02d}.pth"
            torch.save(net.state_dict(), save_path)
            print(f"Model Saved{sr_}")

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Score: {:.4f}".format(best_jaccard))


# ===============================train================================
if __name__ == '__main__':
    for key in cfg:
        print(key, ':', cfg[key])
    params = {}
    for key, value in cfg.items():
        params[key] = value
    for fold in range(int(cfg['n_fold'])):
        print(f'#' * 15)
        print(f'### Fold: {fold}')
        print(f'#' * 15)
        run = wandb.init(project=cfg['project_name'],
                         config={k: v for k, v in params.items()},
                         name=f"fold-{fold}|model-{cfg['model_name']}"
                         )
        train_loader, valid_loader = prepare_loaders()
        model = build_model()
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=float(cfg['lr']), weight_decay=float(cfg['wd']))
        scheduler = fetch_scheduler(optimizer)
        run_training(net=model, optimizers=optimizer, schdulers=scheduler, num_epochs=int(cfg['epochs']))
        run.finish()
