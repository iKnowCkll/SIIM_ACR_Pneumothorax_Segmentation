import cv2

image_path = '/home/chenk/model_train/kaggle/SIIM_ACR_Pneumothorax_Segmentation/train/siimpng/train_png/1.2.276.0.7230010.3.1.4.8323329.309.1517875162.328197.png'
image = cv2.imread(image_path)
print(image.shape)
