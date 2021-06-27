import argparse
import glob
import numpy as np
import os
import shutil
import random

import cv2
from tqdm import tqdm

from facenet_pytorch import MTCNN, extract_face
from PIL import Image, ImageDraw


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='blurring images')

    parser.add_argument('--root_path', required=False, type=str, default='/mldisk/nfs_shared_/js/nsfw/VCDB_2000', help='images path.')
    parser.add_argument('--save_path', required=False, type=str, default='/mldisk/nfs_shared_/js/nsfw/filtered_image/averaging-31-vcdb', help='save path.')
    parser.add_argument('--blur', required=False, type=str, default='averaging', help='1) averaging 2) gaussian 3) median 4) bilateral. 5) mosaic')
    parser.add_argument('--kernel_size', required=False, type=int, default=31, help='Please enter an odd number of 1 or more. If you choose mosaic option, please enter 15 or 30 or 45')
    # kernel size = 11, 21, 31, 41, 51

    args = parser.parse_args()

    root_path = args.root_path
    save_folder_path = args.save_path
    blur = args.blur
    kernel_size = int(args.kernel_size)

    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    detector = MTCNN(keep_all=True)

    images = os.listdir(root_path)

    for img in tqdm(images):
        img_path = os.path.join(root_path, img)
        # save_path = os.path.join(save_folder_path, f'{blur}-{kernel_size}-{img}')
        save_path = os.path.join(save_folder_path, img)

        image = Image.open(img_path)
        cv_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        boxes, probs, points = detector.detect(image, landmarks=True)

        # Success
        if boxes is not None:
            # Success
            for box in boxes:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2] - box[0])
                h = int(box[3] - box[1])

                if x < 0:
                    x = 0
                if y < 0:
                    y = 0

                face = cv_image[y:y+h, x:x+w].copy()

                if blur == 'averaging':
                    face = cv2.blur(face, (kernel_size, kernel_size))
                elif blur == 'gaussian':
                    face = np.clip((face/ 255 + np.random.normal(scale=0.1, size=face.shape)) * 255, 0, 255).astype('uint8')
                    face = cv2.GaussianBlur(face, (kernel_size, kernel_size), 0)
                elif blur == 'median':
                    face = cv2.medianBlur(face, kernel_size)
                elif blur == 'bilateral':
                    face = np.clip((face/ 255 + np.random.normal(scale=0.1, size=face.shape)) * 255, 0, 255).astype('uint8')
                    face = cv2.bilateralFilter(face, kernel_size, 75, 75)
                elif blur == 'mosaic':
                    pass

                cv_image[y:y+h, x:x+w] = face[0:h, 0:w]
            cv2.imwrite(save_path, cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR))
            # # print("save_path :", save_path)
            # # cv2.imwrite(save_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))


    # root_path = '/mldisk/nfs_shared_/MLVD/VCDB-core/frames/core_dataset'
    # save_path = '/mldisk/nfs_shared_/js/nsfw/VCDB_core-2000'
    # file = open('/mldisk/nfs_shared_/js/nsfw/VCDB_core-2000.txt', 'r')
    #
    # for line in tqdm(file.readlines()):
    #     line = " ".join(line.split())
    #     c, v, f = line.split(' ')
    #     shutil.copy(os.path.join(root_path, c, v, f), os.path.join(save_path, f'{c}-{v}-{f}'))

    # for frame in os.listdir(save_folder_path):
    #     f = frame.split('-')[0]
    #     print("f :", f)
    #     if f not in video_name:
    #         video_name.append(f)
    #
    # # print("video_name :", video_name)
    # print("len(video_name) :", len(video_name))
    # parser = argparse.ArgumentParser(description='blurring images')
    #
    # parser.add_argument('--images_path', required=False, default='/mldisk/nfs_shared_/js/nsfw/nsfw_face', help='images path')
    # parser.add_argument('--save_path', required=False, default='/mldisk/nfs_shared_/js/nsfw/facedetect', help='save path')
    # parser.add_argument('--blur', required=False, default='asdf', help='choose one 1, 2, 3,')
    #
    # args = parser.parse_args()
    #
    # root_path = args.images_path
    # save_folder_path = args.save_path
    #
    # detector = MTCNN()
    #
    # f = open('/mldisk/nfs_shared_/js/nsfw/no_face_image.txt', 'w')
    # more_than_one_face = open('/mldisk/nfs_shared_/js/nsfw/more-than-one-face.txt', 'w')
    #
    # for image_name in tqdm(os.listdir(root_path)):
    #     image_path = os.path.join(root_path, image_name)
    #     save_path = os.path.join(save_folder_path, image_name)
    #     image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    #
    #     result = detector.detect_faces(image)
    #     if len(result) != 0:
    #         # Success
    #         if len(result) > 1:
    #             more_than_one_face.write(image_name+'\n')
    #         for i in range(len(result)):
    #             bounding_box = result[i]['box']
    #             cv2.rectangle(image,
    #                           (bounding_box[0], bounding_box[1]),
    #                           (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
    #                           (0, 155, 255),
    #                           2)
    #         cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    #
    #     else:
    #         # Failhtop
    #         f.write(image_name+'\n')
    #
    # more_than_one_face.close()
    # f.close()
    #
