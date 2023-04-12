import albumentations as A
import cv2


class DataTransforms:
    def __init__(self, img_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.img_size = img_size
        self.mean = mean
        self.std = std

    def __call__(self, phase):
        if phase == "train":
            return A.Compose([
                A.HorizontalFlip(),
                A.OneOf([
                    A.RandomContrast(),
                    A.RandomGamma(),
                    A.RandomBrightness(),
                ], p=0.3),
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                    A.GridDistortion(),
                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                ], p=0.3),
                A.ShiftScaleRotate(),
                A.Resize(self.img_size[0], self.img_size[1], interpolation=cv2.INTER_NEAREST),
                A.Normalize(mean=self.mean, std=self.std, p=1)
            ], p=1.0)
        elif phase == "valid":
            return A.Compose([
                A.Resize(self.img_size[0], self.img_size[1], interpolation=cv2.INTER_NEAREST),
                A.Normalize(mean=self.mean, std=self.std, p=1)
            ], p=1.0)
        else:
            raise ValueError("phase should be either 'train' or 'valid'")
