import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def make_transform(args):
    base_transform = [
        A.Resize(args.img_size, args.img_size),
        A.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2471, 0.2435, 0.2616],
        ),
        ToTensorV2()
    ]

    train_transform = []

    if args.RandomBrightnessContrast:
        train_transform.append(
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=0.5))
    if args.HueSaturationValue:
        train_transform.append(A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5))
    if args.RGBShift:
        train_transform.append(A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5))
    if args.RandomGamma:
        train_transform.append(A.RandomGamma(gamma_limit=(80, 120), p=0.5))
    if args.ImageCompression:
        train_transform.append(A.ImageCompression(quality_lower=99, quality_upper=100))
    if args.ShiftScaleRotate:
        train_transform.append(A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7))

    train_transform.extend(base_transform)

    train_transform = A.Compose(train_transform)
    test_transform = A.Compose(base_transform)

    return train_transform, test_transform