import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import joblib
import argparse
import random
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans, KMeans
import os

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

    dim_N = 0   #record the number of patches
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


def vote_for_superpixel(wsi_info_path,save_path,clusterer):
    os.makedirs(save_path,exist_ok=True)
    for slidename in os.listdir(wsi_info_path):
        slide_info={}
        print(slidename)
        slide_info_path=os.path.join(wsi_info_path,slidename)
        wsi_info=torch.load(slide_info_path)
        superpixels=[]
        for index in range(len(wsi_info)):
            superpixels.append(wsi_info[index]['superpixel'])
        
        for superpixel_value in range(1,max(superpixels)+1):   #begin from 1
            cluster_num=np.zeros(100,dtype=int)
            superpixel_info={}
#             slide_info[superpixel_value]['feature']=superpixel_fea[superpixel_value]
            for index in range(len(wsi_info)):
#                 print(type(wsi_info[index]['superpixel']))
                if wsi_info[index]['superpixel']==superpixel_value:
                    patch_fea=wsi_info[index]['features'].reshape(1,-1)  
#                     cluster=wsi_info[index]['cluster']
                    cluster=clusterer.predict(patch_fea)[0]
#                     print('cluster:{}'.format(cluster))
                    cluster_num[cluster]+=1
            superpixel_cluster=np.argmax(cluster_num)  #vote
#             slide_info[superpixel_value]['cluster']=superpixel_cluster
            superpixel_info['cluster']=superpixel_cluster
            slide_info[superpixel_value]=superpixel_info
        slide_info_path=os.path.join(save_path,slidename)
        torch.save(slide_info,slide_info_path)

def main(args):
    cluster_data = torch.load(args.cluster_info_dir)
    print(len(cluster_data))    
    train_data = []
    for data in cluster_data:
        train_data.append(data)     
    print(len(train_data))
    clusterer = get_cluster(train_data, k=args.vw_num, num=-1, seed=3)  
    wsi_info_path=args.wsi_info_path
    save_path= args.save_path
    vote_for_superpixel(wsi_info_path,save_path,clusterer)


def get_params():
    parser = argparse.ArgumentParser(description='superpixel cluster')

    parser.add_argument('--cluster_info_dir', type=str, default='/data14/yanhe/miccai/codebook/data/argo/codebook_info.pt')
    parser.add_argument('--vw_num', type=int, default=16)
    parser.add_argument('--wsi_info_path', type=str, default='/data14/yanhe/miccai/super_pixel/slide_superpixel/argo_new/superpixel_num_600/')
    parser.add_argument('--save_path',type=str, default='/data14/yanhe/miccai/codebook/cluster_info/argo/superpixel600_cluster16/all_fold/')

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
#         tuner_params = nni.get_next_parameter()
#         logger.debug(tuner_params)
#         params = vars(merge_parameter(get_params(), tuner_params))
#         main(params)
        args=get_params()
        main(args)
    except Exception as exception:
#         logger.exception(exception)
        raise

