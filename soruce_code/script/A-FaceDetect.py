import argparse
import glob
import os

import cv2
from tqdm import tqdm

from facenet_pytorch import MTCNN, extract_face
from PIL import Image, ImageDraw

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Face Detect')

    parser.add_argument('--root_path', required=False, type=str, default='/mldisk/nfs_shared_/js/nsfw/nsfw_face_detect_1179', help='images path')
    parser.add_argument('--save_path', required=False, type=str, default='/mldisk/nfs_shared_/js/nsfw/nsfw_1149', help='save path')
    parser.add_argument('--face_draw', required=False, type=bool, default=False, help='choose True or False')

    args = parser.parse_args()

    root_path = args.root_path
    save_folder_path = args.save_path
    face_draw = args.face_draw

    detector = MTCNN(keep_all=True)

    images = os.listdir(root_path)

    for img in tqdm(images):
        img_path = os.path.join(root_path, img)
        save_path = os.path.join(save_folder_path, img)

        image = Image.open(img_path)

        boxes, probs, points = detector.detect(image, landmarks=True)

        # Face Detect Success
        if boxes is not None:
            if face_draw:
                img_draw = image.copy()
                draw = ImageDraw.Draw(img_draw)
                for i, box in enumerate(boxes):
                    draw.rectangle(box.tolist(), width=2)
                # extract_face(image, box, save_path=save_path)
                img_draw.save(save_path)
            else:
                image.save(save_path)