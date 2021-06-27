import subprocess
import os

# original_paths = ['/mldisk/nfs_shared_/js/nsfw/VCDB_2000', '/mldisk/nfs_shared_/js/nsfw/nsfw_face_detect_1179']
original_paths = ['/mldisk/nfs_shared_/js/inverse/vcdb_50000_up']
blur = ['averaging', 'gaussian', 'median']
kernel_size = [41, 51, 61]

for original_path in original_paths:
    for b in blur:
        for k in kernel_size:
            root_path = original_path
            save_path = f'/mldisk/nfs_shared_/js/inverse/blurring_image/{b}-{k}'
            command = f'/opt/conda/bin/python -u B-BlurringImage.py --root_path {root_path} --save_path {save_path} --blur {b} --kernel_size {k}'
            print(f"dataset : {original_path.split('/')[-1]}, blur : {b}, kernel_size : {k}")
            subprocess.call(command, shell=True)
