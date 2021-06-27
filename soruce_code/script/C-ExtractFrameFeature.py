import io

from torchvision.transforms import transforms as trn
from torch.utils.data import DataLoader, Dataset
import torch

import cv2
from tqdm import tqdm
import numpy as np
import argparse
import os

from PIL import Image, ImageDraw

from VCD.utils import DEVICE_STATUS, DEVICE_COUNT
from VCD import models
from VCD import datasets


class ImageListDataset(Dataset):
    def __init__(self, frames, root_dir, transform=None):
        self.frames = frames
        self.root_dir = root_dir

        default_transform = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform = transform or default_transform

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.frames[idx])
        image = Image.open(path)
        return self.transform(image), self.frames[idx]

    def __len__(self):
        return len(self.frames)

    def __repr__(self):
        fmt_str = f'{self.__class__.__name__}\n'
        fmt_str += f'\tNumber of images : {self.__len__()}\n'
        trn_str = self.transform.__repr__().replace('\n', '\n\t')
        fmt_str += f"\tTransform : \n\t{trn_str}"

        return fmt_str

@torch.no_grad()
def extract_frame_features(model, loader, frames, root_path, save_to):
    model.eval()

    loader.dataset.l = frames
    bar = tqdm(loader, unit='batch')
    features = []

    for idx, (images, paths) in enumerate(bar):
        # print(f"{images}, {paths}")
        # print("")
        feat = model(images.cuda()).cpu()
        torch.save(feat, os.path.join(save_to, f'{paths[0]}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frame feature')
    parser.add_argument('--model', required=False, type=str, default='Resnet50_RMAC', help='Resnet50_RMAC')
    parser.add_argument('--root_path', required=False, type=str, default='/workspace/image', help='images path.')
    parser.add_argument('--feature_path', required=False, type=str, default='/workspace/feature', help='images path.')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--worker', type=int, default=4)

    args = parser.parse_args()

    if os.path.exists(args.feature_path) and len(os.listdir(args.feature_path)) != 0:
        print(f'Feature Path {args.feature_path} is not empty.')
        exit(1)

    if not os.path.exists(args.feature_path):
        os.mkdir(args.feature_path)

    root_path = args.root_path

    # models
    model = models.get_frame_model(args.model).cuda()

    # Check device
    print("DEVICE_STATUS :", DEVICE_STATUS)
    print("DEVICE_COUNT :", DEVICE_COUNT)
    if DEVICE_STATUS and DEVICE_COUNT > 1:
        model = torch.nn.DataParallel(model)

    transform = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset
    frames = os.listdir(root_path)
    face_dataset = ImageListDataset(frames, root_path, transform)

    loader = DataLoader(face_dataset, batch_size=args.batch, shuffle=False, num_workers=args.worker)

    extract_frame_features(model, loader, frames, root_path, args.feature_path)