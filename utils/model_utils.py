from configparser import ConfigParser

import segmentation_models_pytorch as smp
import torch

config_parser = ConfigParser()
config_parser.read('/home/chenk/model_train/kaggle/SIIM_ACR_Pneumothorax_Segmentation/mysql.cfg')
cfg = config_parser['default']

checkpoints_path = cfg['checkpoint_path']
checkpoints_weights = torch.load(checkpoints_path)


def build_model():
    model = smp.Unet(
        encoder_name=cfg['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=int(cfg['num_classes']),  # model output channels (number of classes in your dataset)
        activation=None,
    )
    if int(cfg['checkpoint']) == 1:
        model.load_state_dict(checkpoints_weights)
        print('load checkpoints weights')
    else:
        print('load pretrained weights')
    return model

