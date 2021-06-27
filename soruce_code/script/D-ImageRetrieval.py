import argparse
from tqdm import tqdm
import numpy as np
import faiss
import torch
import os


def load(path):
    feat = torch.load(path)
    return feat


def load_features(images, feature_root):
    features = []
    for img in tqdm(images):
        path = os.path.join(feature_root, f'{img}')
        features.append(load(path))

    length = [f.shape[0] for f in features]

    start = np.cumsum([0] + length)
    index = np.concatenate((start[:-1].reshape(-1, 1), start[1:].reshape(-1, 1)), axis=1)
    images = {v: n for n, v in enumerate(images)}

    return np.concatenate(features), images, index


if __name__ == '__main__':
    print("image retrieval")

    parser = argparse.ArgumentParser(description='Extract frame feature')
    # parser.add_argument('--image_path', required=False, type=str, default='/mldisk/nfs_shared_/js/nsfw/VCDB_1878', help='Database image path')
    parser.add_argument('--database_path', required=False, type=str, default='/mldisk/nfs_shared_/js/nsfw/features/original/nsfw_1149', help='Database feature path')
    parser.add_argument('--feature_path', required=False, type=str, default='/mldisk/nfs_shared_/js/nsfw/features/filtering/nsfw_face_detect_1179/median-61', help='query feature directory path')
    parser.add_argument('--topK', required=False, type=int, default=1, help='topK')
    parser.add_argument('--excel_path', required=False, type=str, default='/mldisk/nfs_shared_/js/nsfw/excel/averaging-21-nsfw.csv', help='result excel file')
    args = parser.parse_args()

    database_path = args.database_path
    feature_path = args.feature_path
    topK = args.topK
    excel_path = args.excel_path

    # Dataset
    dataset = []
    for img in os.listdir(database_path):
        dataset.append(img)
    dataset = np.array(dataset)

    # Database features
    features, images, loc = load_features(dataset, database_path)

    table = dict()
    count = 0

    for image_idx, ran in enumerate(loc):
        for features_idx in range(ran[1] - ran[0]):
            table[count] = (image_idx, features_idx)
            count += 1

    mapping = np.vectorize(lambda x, table: table[x])

    index = faiss.IndexFlatIP(features.shape[1])
    faiss.normalize_L2(features)
    index.add(features)

    print("\nCalculating results. . . .")

    bar = tqdm(dataset, mininterval=0.5, ncols=150)

    find_count = 0
    rank_sum = 0
    rank_list = []

    for query_idx, query_image_name in enumerate(bar):
        query_path = os.path.join(feature_path, f'{query_image_name}')
        query_feat = load(query_path).numpy()
        D, I = index.search(query_feat, topK)

        topK_idx = mapping(I, table)[0][0]

        # topK list
        rank = -1
        for rank_idx, result_idx in enumerate(topK_idx):
            if query_idx == result_idx:
                rank = rank_idx + 1

        # if rank == 1 : not detect
        if rank != -1:
            find_count += 1
            rank_sum += rank
            rank_list.append(rank)

    rank_not_1 = len([q for q in rank_list if q != 1])

    print(f'success percent : {format(find_count/len(dataset)*100, ".2f")}, ' 
          f'averaging ranking : {format(rank_sum/find_count, ".2f")}, '
          f'recall : {format(find_count/len(dataset), ".2f")}, '
          f'rank_not_1/len(rank_list) : {rank_not_1}/{len(rank_list)}')
