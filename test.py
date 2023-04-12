from configparser import ConfigParser
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from my_dataset import TestDataset
from utils.data_utils import run_length_encode
from utils.model_utils import build_model

config_parser = ConfigParser()
config_parser.read('/home/chenk/model_train/kaggle/SIIM_ACR_Pneumothorax_Segmentation/mysql.cfg')
cfg = config_parser['default']

size = 512
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
num_workers = 8
batch_size = 16
device = torch.device("cuda:0")
thre = [0.75, 600, 0.5]
threshold = 0.5
min_size = 4000
df = pd.read_csv('/home/chenk/model_train/kaggle/SIIM_ACR_Pneumothorax_Segmentation/stage_2_sample_submission.csv')
test_data_folder = "/home/chenk/model_train/kaggle/SIIM_ACR_Pneumothorax_Segmentation/test"
test_set = TestDataset(test_data_folder, df, size, mean, std)
test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

model = build_model()
model.to(device)
model.eval()
encoded_pixels = []


def post_process(probability, threshold, min_size):
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((1024, 1024), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num

# def post_process(pre_mask, thre):
#     classification_mask = pre_mask > thre[0]
#     total_true = np.sum(classification_mask)
#     mask = pre_mask.copy()
#     if total_true < thre[1]:
#         num = 0
#     else:
#         mask = (mask > thre[2])
#         mask = np.array(mask, dtype=np.float32)
#         num = 1
#     return mask, num


for i, batch in enumerate(tqdm(test_loader)):
    preds = []
    for fold in range(int(cfg['n_fold'])):
        model.eval()
        state = torch.load(f'/home/chenk/model_train/kaggle/SIIM_ACR_Pneumothorax_Segmentation'
                           f'/save_model/best_stage/best_epoch-0{fold}.pth')
        model.load_state_dict(state)
        with torch.no_grad():
            pred = torch.sigmoid(model(batch.to(device)))
            pred = pred.detach().cpu().numpy()[:, 0, :, :]  # (batch_size, 1, size, size) -> (batch_size, size, size)
            preds.append(pred)
    preds = np.mean(preds, axis=0)

    for probability in preds:
        if probability.shape != (1024, 1024):
            probability = cv2.resize(probability, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
        predict, num_pred = post_process(probability, threshold=threshold, min_size=min_size)
        if num_pred == 0:
            encoded_pixels.append('-1')
        else:
            r = run_length_encode(predict)
            encoded_pixels.append(r)

df['EncodedPixels'] = encoded_pixels
df.to_csv('submission7.csv', columns=['ImageId', 'EncodedPixels'], index=False)
