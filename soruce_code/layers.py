import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms as trn

from PIL import Image, ImageDraw
import numpy as np
import cv2


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU()
        self.maxPooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)

        self.deconv = nn.ConvTranspose2d(3, 64, kernel_size=3, stride=3, padding=58, bias=False)
        self.unpooling = nn.MaxUnpool2d(kernel_size=4, stride=2, padding=1)

        self.deconv.weight = self.conv1.weight
        print("conv1 weight :", self.conv1.weight)
        print("decon weight :", self.deconv.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out, indices = self.maxPooling(out)

        return out, indices

    # def test(self, x):
    #     out = self.conv1(x)
    #     out = self.relu(out)
    #     return out

    def conv_forward(self, x):
        out = self.conv1(x)
        return out

    def maxpooling_inverse(self, x, indices):
        unpooling_x = self.unpooling(x, indices)
        return unpooling_x

    def conv1_inverse(self, x):
        deconv_x = self.deconv(x)
        return deconv_x

    # 3, 224, 224
    def inverse2(self, x, indices):
        out = self.maxpooling_inverse(x)
        out = self.relu_inverse(out)
        out = self.conv1_inverse(out)

        return out


def save_image_tensor2cv2(input_tensor: torch.Tensor, image_size, filename):
    """
    Save tensor to cv2 format
         :param input_tensor: tensor to save
         :param filename: saved file name
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
         # Make a copy
    input_tensor = input_tensor.clone().detach()
         # To cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
         # Denormalization
    # input_tensor = unnormalize(input_tensor)
         # Remove batch dimension
    input_tensor = input_tensor.squeeze()
         # Convert from [0,1] to [0,255], then from CHW to HWC, and finally to cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
         # RGB to BRG
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    input_tensor = cv2.resize(input_tensor, dsize=image_size)
    cv2.imwrite(filename, input_tensor)


def main():
    transform = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    nn = Network()
    model = nn.cuda()

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    path = "/workspace/0000##--4ziAJuyZ0.mp4##000023.jpg"
    image = Image.open(path)
    original_size = image.size

    model.eval()
    image = torch.from_numpy(np.expand_dims(transform(image), axis=0))

    feat, indices = model(image)

    # max pooling inverse
    unpooling = model.module.maxpooling_inverse(feat, indices)
    # save_image_tensor2cv2(unpooling, original_size, "/workspace/image/unpooling.jpg")

    # print("unpooling :", unpooling)
    # print("unpooling.shape :", unpooling.shape)

    # convolution inverse (create image that has (3, 224, 224) size)
    deconv = model.module.conv1_inverse(unpooling)

    # print("deconv.shape :", deconv.shape)
    # print("transform image shape :", image.shape)
    # print("feature.shape :", feat.shape)

    # (3, 224, 224) ==> original image
    save_image_tensor2cv2(deconv, original_size, "/workspace/image/inverse.jpg")


if __name__ == '__main__':
    main()