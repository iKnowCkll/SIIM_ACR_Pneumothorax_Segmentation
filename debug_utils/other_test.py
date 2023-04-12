from configparser import ConfigParser

import cv2
import numpy as np
import pandas as pd
import torch
from albumentations import Compose, Normalize, Resize
from matplotlib import pyplot as plt

from utils.data_utils import run_length_decode
from utils.model_utils import build_model

# config_parser = ConfigParser()
# config_parser.read('/home/chenk/model_train/kaggle/SIIM_ACR_Pneumothorax_Segmentation/mysql.cfg')
# cfg = config_parser['default']
#
# data_path = '/home/chenk/model_train/kaggle/SIIM_ACR_Pneumothorax_Segmentation/train/siimpng/train_png'
# df_path = '/home/chenk/model_train/kaggle/SIIM_ACR_Pneumothorax_Segmentation/' \
#           'siim_acr_pneumothorax_segmentation/train-rle.csv'
# df_all = pd.read_csv(df_path)
# df = df_all.drop_duplicates('ImageId')
# df.loc[df[" EncodedPixels"] != "-1", 'has_mask'] = 1
# df.loc[df[" EncodedPixels"] == "-1", 'has_mask'] = 0
# df_with_mask = df[df[" EncodedPixels"] != "-1"]
# df_without_mask = df[df[" EncodedPixels"] == "-1"]
# df_without_mask_sampled = df_without_mask.sample(len(df_with_mask) + 1500, random_state=101)  # random state is imp
# df = pd.concat([df_with_mask, df_without_mask_sampled]).reset_index(drop=True)
#
# images_rle_mask = df[' EncodedPixels'].tolist()
# rle_mask = images_rle_mask[0]
# mask = np.zeros([1024, 1024])
# mask += run_length_decode(rle_mask)
# print(mask.shape)
config_parser = ConfigParser()
config_parser.read('/home/chenk/model_train/kaggle/SIIM_ACR_Pneumothorax_Segmentation/mysql.cfg')
cfg = config_parser['default']

size = 512
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
num_workers = 8
batch_size = 16
best_threshold = 0.5
min_size = 3500
device = torch.device("cuda:0")

model = build_model()
model.to(device)

state = torch.load('/home/chenk/model_train/kaggle/SIIM_ACR_Pneumothorax_Segmentation'
                   '/save_model/stage0/best_epoch-00.pth')
model.load_state_dict(state)

image_path = '/home/chenk/model_train/kaggle/SIIM_ACR_Pneumothorax_Segmentation/train/siimpng/train_png' \
             '/1.2.276.0.7230010.3.1.4.8323329.13666.1517875247.117800.png'

image = cv2.imread(image_path)
transform = Compose(
    [
        Normalize(mean=mean, std=std, p=1),
        Resize(size, size)
    ])
images = transform(image=image)["image"]
images = np.transpose(images, (2, 0, 1))
images = torch.from_numpy(images).float()
images = images.unsqueeze(0)
print(images.shape)

preds = model(images.to(device))
preds = torch.sigmoid(preds)
y_pred = (preds > 0.5).to(torch.float32)
y_pred = y_pred.squeeze(0).squeeze(0).cpu().detach().numpy()
y_pred = cv2.resize(y_pred, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)

# plt.imshow(y_pred)
#
# # 提取像素值为1的像素点的索引
# indices = np.where(y_pred == 1)
#
# # 将像素点标记为红色
# plt.plot(indices[1], indices[0], 'ro', markersize=2)
#
# # 显示图片
# plt.show()
# print(y_pred.shape)
#
# rel='557374 2 1015 8 1009 14 1002 20 997 26 990 32 985 38 980 42 981 42 979 43 978 45 976 47 964 59 956 66 925 98 922 101 917 106 916 106 916 107 914 109 909 113 907 116 904 118 903 120 902 120 902 121 900 122 899 124 898 124 898 125 897 125 898 125 896 126 895 127 895 128 895 128 895 128 894 128 895 128 895 128 895 128 895 128 895 128 894 130 893 130 893 130 893 130 893 129 894 129 894 129 894 129 895 127 897 126 898 126 898 125 898 126 898 125 899 125 899 125 899 124 900 124 900 125 899 125 899 125 899 125 899 126 898 127 897 128 897 128 896 129 895 130 894 132 892 133 891 134 890 136 888 137 888 137 887 139 885 140 884 141 884 141 883 142 882 143 881 145 879 146 879 147 878 147 877 149 876 150 874 153 872 153 871 154 871 154 870 155 870 154 870 154 870 154 871 153 871 154 871 154 870 154 871 154 870 155 869 155 870 155 869 155 869 155 869 155 869 156 869 155 870 154 871 153 872 153 872 152 872 152 873 152 872 152 873 152 872 152 872 153 872 152 872 153 872 152 872 152 873 151 874 151 874 150 875 150 874 150 875 150 874 150 874 151 873 152 872 153 872 152 872 153 871 153 872 153 872 152 873 151 874 151 874 150 875 150 874 150 875 149 876 149 875 150 875 149 875 150 875 150 875 149 876 149 876 149 876 149 876 149 876 149 876 148 877 148 878 146 880 144 881 144 881 143 882 142 883 142 883 142 882 143 882 143 882 142 883 142 883 141 884 141 884 140 885 140 885 140 885 140 885 140 886 139 886 139 887 138 887 138 888 137 890 135 891 135 892 133 894 131 894 131 894 131 894 130 896 129 896 129 897 127 899 126 900 125 901 126 899 126 900 125 901 124 901 124 902 123 903 122 904 121 905 120 906 119 906 119 906 119 907 118 907 118 908 118 908 118 909 116 910 115 912 113 914 110 916 109 918 107 919 105 921 104 923 101 925 100 925 99 926 99 927 98 927 97 929 96 930 95 932 93 933 92 935 90 938 87 940 85 941 84 943 82 943 82 944 80 946 79 947 77 948 77 949 75 951 73 955 70 957 68 957 68 958 67 959 65 961 63 963 61 966 58 968 56 971 53 975 48 981 37 992 27'
# mask = np.zeros([1024, 1024])
# mask += run_length_decode(rel)
def post_process(pre_mask, threshold):
    classification_mask = pre_mask > threshold[0]
    total_true = np.sum(classification_mask)
    mask = pre_mask.copy()
    if total_true < threshold[1]:
        num = 0
    else:
        mask = (mask > threshold[2])
        mask = np.array(mask, dtype=np.float32)
        num = 1
    return mask, num
#
# def post_process(probability, threshold, min_size):
#     mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
#     num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
#     predictions = np.zeros((1024, 1024), np.float32)
#     num = 0
#     for c in range(1, num_component):
#         p = (component == c)
#         if p.sum() > min_size:
#             predictions[p] = 1
#             num += 1
#     return predictions, num


mask, num = post_process(y_pred, threshold=(0.75, 2000, 0.4))
print(num)
plt.imshow(mask)

# 提取像素值为1的像素点的索引
indices = np.where(mask == 1)

# 将像素点标记为红色
plt.plot(indices[1], indices[0], 'ro', markersize=2)

# 显示图片
plt.show()

