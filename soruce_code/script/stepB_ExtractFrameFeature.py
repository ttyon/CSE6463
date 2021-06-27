import subprocess
import os


# root_paths = ['/mldisk/nfs_shared_/js/nsfw/filtered_image/nsfw_face_detect_1179']
# root_paths = ['/mldisk/nfs_shared_/js/inverse/blurring_image', '/mldisk/nfs_shared_/js/nsfw/filtered_image/nsfw_face_detect_1179']
root_paths = ['/mldisk/nfs_shared_/js/inverse/blurring_image']
blur = ['averaging', 'gaussian', 'median']
kernel_size = [41, 51, 61]
model = 'Resnet50_RMAC'

for root in root_paths:
    for b in blur:
        for k in kernel_size:
            root_path = os.path.join(root, f'{b}-{k}')
            feature_path = f'/mldisk/nfs_shared_/js/inverse/feature/{b}-{k}'

            command = f'/opt/conda/bin/python -u C-ExtractFrameFeature.py --model {model} --root_path {root_path} --feature_path {feature_path}'
            print(command)
            subprocess.call(command, shell=True)


command = f'/opt/conda/bin/python -u C-ExtractFrameFeature.py --model {model} --root_path /mldisk/nfs_shared_/js/inverse/vcdb_50000_up --feature_path /mldisk/nfs_shared_/js/inverse/feature/original'
print(command)
subprocess.call(command, shell=True)