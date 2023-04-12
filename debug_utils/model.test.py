import segmentation_models_pytorch as smp
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_model():
    model = smp.Unet(
        encoder_name='se_resnext50_32x4d',  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
        activation=None,
    )
    model.to(device)
    return model


model = build_model()
print(model)

input = torch.randn(1, 3, 512, 512).to(device)
output = model(input)
print(output.shape)
