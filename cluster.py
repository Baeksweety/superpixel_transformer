import torch 
import numpy as np
import os
import h5py
from sklearn.cluster import MiniBatchKMeans, KMeans
import os
import random
import argparse
# 3rd party library
from tqdm import tqdm
import numpy as np
import pandas as pds
from sklearn.cluster import MiniBatchKMeans, KMeans
import torch

def get_cluster(datas, k=100, num=-1, seed=3):
    if seed is not None:
        random.seed(seed)
    if num > 0:
        data_sam = random.sample(datas, num)
    else:
        data_sam = datas
    if seed is not None:
        random.seed(seed)
    random.shuffle(data_sam)

    dim_N = 0   #record the number of all patches
    dim_D = data_sam[0]['shape'][1]
    for data in tqdm(data_sam):
        dim_N += data['shape'][0]
    con_data = np.zeros((dim_N, dim_D), dtype=np.float32)
    ind = 0
    for data in tqdm(data_sam):
        data_path, data_shape = data['slide'], data['shape']
        cur_data = torch.load(data_path)
        con_data[ind:ind + data_shape[0], :] = cur_data.numpy()
        ind += data_shape[0]
    # clusterer = KMeans(n_clusters=k)
    clusterer = MiniBatchKMeans(n_clusters=k, batch_size=10000)
    clusterer.fit(con_data)
    print("cluster done")
    return clusterer

for fold_num in range(5):
    cluster_data = torch.load('/data14/yanhe/miccai/codebook/data/argo/codebook_info_fold{}.pt'.format(fold_num))
#     print(len(cluster_data))
    train_data = []
    for data in cluster_data:
        train_data.append(data)
    print(len(train_data))
    clusterer = get_cluster(train_data, k=16, num=-1, seed=3)   #k represents the number of clusters
    saved_path='/data14/yanhe/miccai/codebook/patch_cluster/argo/fold{}/'.format(fold_num)
    os.makedirs(saved_path,exist_ok=True)
    for slidename in os.listdir('/data12/ybj/survival/argo_selected/20x/slides_feat/h5_files'):
        if slidename[-3:] == '.h5':
            slide_path=os.path.join('/data12/ybj/survival/argo_selected/20x/slides_feat/h5_files',slidename)
#             print(slide_path)
            f = h5py.File(slide_path)
            name, _ = os.path.splitext(slidename)
            print(name)
            slide_cluster_path=saved_path+name+'.pth'
#             print(slide_cluster_path)
            length = f['coords'].shape[0]
            wsi_patch=[]
            for i in range(length):
                wsi_patch_info = {}
                patch_fea = f['features'][i].reshape(1,-1)
                cluster_class = clusterer.predict(patch_fea)[0]
                wsi_patch_info['patch_id']=i
                wsi_patch_info['cluster']=cluster_class
                wsi_patch.append(wsi_patch_info)
            torch.save(wsi_patch,slide_cluster_path)
