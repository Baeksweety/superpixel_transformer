import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd
import h5py
from sklearn.cluster import MiniBatchKMeans, KMeans
import random 
from PIL import Image
from dgl.data.utils import save_graphs

from histocartography.utils import download_example_data
from histocartography.preprocessing import (
    ColorMergedSuperpixelExtractor,
    DeepFeatureExtractor
)

from histocartography.visualization import OverlayGraphVisualization
from superpixel_utils import RAGGraphBuilder,MyColorMergedSuperpixelExtractor
import argparse
from skimage.measure import regionprops
import joblib
import cv2


#remove background
def get_node_centroids(instance_map: np.ndarray,raw_image:np.ndarray):
#     print(instance_map)
    regions = regionprops(instance_map)
    mask_value=1
#     centroids = np.empty((len(regions), 2))
    for i, region in enumerate(regions):
        center_y, center_x = region.centroid  # (y, x)  
#         print(i)
#         print(region.coords)
        center_x = int(round(center_x))
        center_y = int(round(center_y))
        if sum(raw_image[center_y,center_x])== 0:
            for index,couple in enumerate(region.coords):
                y,x = couple
#                 print(y,x)
                instance_map[y,x]=0
        else:
            for index,couple in enumerate(region.coords):
                y,x = couple
#                 print(y,x)
                instance_map[y,x]=mask_value
            mask_value+=1

#         centroids[i, 0] = center_x
#         centroids[i, 1] = center_y
#         print(instance_map)
    return instance_map

def generate_superpixel(image_path,downsampling_factor):
    """
    Generate a tissue graph for all the images in image path dir.
    """

    # 1. get image path
#     image_fnames = glob(os.path.join(image_path, '*.png'))
#     image_fnames = img_path
    # 2. define superpixel extractor. Here, we query 50 SLIC superpixels,
    # but a superpixel size (in #pixels) can be provided as well in the case
    # where image size vary from one sample to another.
    superpixel_detector = ColorMergedSuperpixelExtractor(
        nr_superpixels=args.nr_superpixels,
        compactness=10,
        blur_kernel_size=1,
        threshold=1,
        downsampling_factor=downsampling_factor,
        connectivity = 2,
    )

    # 3. define feature extractor: extract patches of 144x144 pixels
    # resized to 224 to match resnet input size. If the superpixel is larger
    # than 144x144, several patches are extracted and patch embeddings are averaged.
    # Everything is handled internally. Please refer to the implementation for
    # details.
    feature_extractor = DeepFeatureExtractor(
        architecture='resnet34',
        patch_size=144,
        resize_size=224
    )

    # 4. define graph builder
    tissue_graph_builder = RAGGraphBuilder(add_loc_feats=True)

    # 5. define graph visualizer
    visualizer = OverlayGraphVisualization()

    _, image_name = os.path.split(image_path)
    image = Image.open(image_path)
    image=np.array(image)
    fname = image_name[:-4]+'.png'
    print(image.shape)

    superpixels, _ = superpixel_detector.process(image)
    superpixels = get_node_centroids(superpixels,image)
    return superpixels


def modify_superpixels(superpixel_patchnum,instance_map):
    regions = regionprops(instance_map)
    mask_value=1
    for i, region in enumerate(regions):
        sup_value = region.label
        if (superpixel_patchnum[sup_value]==0) or (superpixel_patchnum[sup_value]==1):
            print('true!')
            for index,couple in enumerate(region.coords):
                y,x = couple
                instance_map[y,x]=0
        else:
            for index,couple in enumerate(region.coords):
                y,x = couple
                instance_map[y,x]=mask_value
            mask_value+=1
    return instance_map



def get_max_value(data_matrix):
    new_data=[]
    for i in range(len(data_matrix)):
        new_data.append(max(data_matrix[i]))
    return max(new_data)



def patch_judge_40x(stitch_path,saved_path):
    _, image_name = os.path.split(stitch_path)
    name = image_name[:-4]
    superpixels=generate_superpixel(stitch_path,downsampling_factor=4)
    print(superpixels.shape)
    h,w=superpixels.shape
    slide_info_path = '/data12/ybj/survival/CRC/40x/slides_feat/h5_files/'+name+'.h5'
    print(slide_info_path)
    f = h5py.File(slide_info_path)
    new_slide_info=[]
    new_slide_info_path=saved_path+name+'.pth'
    length = f['coords'].shape[0]
    for i in range(length):
        patch_info={}
        patch_info['patch_id']=i
        patch_info['coords']=f['coords'][i]
        patch_info['features']=f['features'][i]
        patch_coords=f['coords'][i]
#         x,y=slide_info[i]['coords']
        x,y=patch_coords 
        x=x+512
        y=y+512   
        x=int(x/16)   
        y=int(y/16)
        if (y<h)&(x<w):
            patch_info['superpixel']=superpixels[y][x]
        elif (y>=h)&(x<w):
            x_n,y_n=f['coords'][i]
            x_n=x_n+512
            y_n=y_n+(h*16-y_n)/2
            x_n=int(x_n/16)
            y_n=int(y_n/16)
            patch_info['superpixel']=superpixels[y_n][x_n]
        elif (y<h)&(x>=w):
            x_n,y_n=f['coords'][i]
            x_n=x_n+(w*16-x_n)/2
            y_n=y_n+512
            x_n=int(x_n/16)
            y_n=int(y_n/16)
            patch_info['superpixel']=superpixels[y_n][x_n]
        else:
            x_n,y_n=f['coords'][i]
            x_n=x_n+(w*16-x_n)/2
            y_n=y_n+(h*16-y_n)/2
            x_n=int(x_n/16)
            y_n=int(y_n/16)
            patch_info['superpixel']=superpixels[y_n][x_n]
        new_slide_info.append(patch_info)
    superpixel_patchnum={}
    max_superpixel = get_max_value(superpixels)
#     print(max(superpixel))
    print(max_superpixel)
    for sup_value in range(1,max_superpixel+1):   
#             print(sup_value)
        coords_s=[]
        features_s=[]
        for patch_id in range(len(new_slide_info)):
            if new_slide_info[patch_id]['superpixel']==sup_value:
                coords_s.append(new_slide_info[patch_id]['coords'])
#                 features_s.append(new_slide_info[patch_id]['features'])
        coords_np=np.array(coords_s)
#         features_np=np.array(features_s)
#             print(coords_np.shape,features_np.shape)
        patch_number=coords_np.shape[0]
        superpixel_patchnum[sup_value] = patch_number
    superpixels = modify_superpixels(superpixel_patchnum,superpixels)
    wsi_info = []
    for i in range(length):
        patch_info={}
        patch_info['patch_id']=i
        patch_info['coords']=f['coords'][i]
        patch_info['features']=f['features'][i]
        patch_coords=f['coords'][i]
#         x,y=slide_info[i]['coords']
        x,y=patch_coords 
        x=x+512
        y=y+512   
        x=int(x/16)   
        y=int(y/16)
        if (y<h)&(x<w):
            patch_info['superpixel']=superpixels[y][x]
        elif (y>=h)&(x<w):
            x_n,y_n=f['coords'][i]
            x_n=x_n+512
            y_n=y_n+(h*16-y_n)/2
            x_n=int(x_n/16)
            y_n=int(y_n/16)
            patch_info['superpixel']=superpixels[y_n][x_n]
        elif (y<h)&(x>=w):
            x_n,y_n=f['coords'][i]
            x_n=x_n+(w*16-x_n)/2
            y_n=y_n+512
            x_n=int(x_n/16)
            y_n=int(y_n/16)
            patch_info['superpixel']=superpixels[y_n][x_n]
        else:
            x_n,y_n=f['coords'][i]
            x_n=x_n+(w*16-x_n)/2
            y_n=y_n+(h*16-y_n)/2
            x_n=int(x_n/16)
            y_n=int(y_n/16)
            patch_info['superpixel']=superpixels[y_n][x_n]
        wsi_info.append(patch_info)
    torch.save(wsi_info,new_slide_info_path)
    return superpixels    


def patch_judge_20x(stitch_path,saved_path):
    _, image_name = os.path.split(stitch_path)
    name = image_name[:-4]
    superpixels=generate_superpixel(stitch_path,downsampling_factor=2)
    print(superpixels.shape)
    h,w=superpixels.shape
    slide_info_path = '/data12/ybj/survival/CRC/20x/slides_feat/h5_files/'+name+'.h5'
    f = h5py.File(slide_info_path)
    new_slide_info=[]
    new_slide_info_path=saved_path+name+'.pth'
    length = f['coords'].shape[0]
    for i in range(length):
        patch_info={}
        patch_info['patch_id']=i
        patch_info['coords']=f['coords'][i]
        patch_info['features']=f['features'][i]
        patch_coords=f['coords'][i]
        x,y=patch_coords 
        x=x+256
        y=y+256   
        x=int(x/16)   
        y=int(y/16)
        if (y<h)&(x<w):
            patch_info['superpixel']=superpixels[y][x]
        elif (y>=h)&(x<w):
            x_n,y_n=f['coords'][i]
            x_n=x_n+256
            y_n=y_n+(h*16-y_n)/2
            x_n=int(x_n/16)
            y_n=int(y_n/16)
            patch_info['superpixel']=superpixels[y_n][x_n]
        elif (y<h)&(x>=w):
            x_n,y_n=f['coords'][i]
            x_n=x_n+(w*16-x_n)/2
            y_n=y_n+256
            x_n=int(x_n/16)
            y_n=int(y_n/16)
            patch_info['superpixel']=superpixels[y_n][x_n]
        else:
            x_n,y_n=f['coords'][i]
            x_n=x_n+(w*16-x_n)/2
            y_n=y_n+(h*16-y_n)/2
            x_n=int(x_n/16)
            y_n=int(y_n/16)
            patch_info['superpixel']=superpixels[y_n][x_n]
        new_slide_info.append(patch_info)
    superpixel_patchnum={}
    max_superpixel = get_max_value(superpixels)
#     print(max(superpixel))
    print(max_superpixel)
    for sup_value in range(1,max_superpixel+1):   
#             print(sup_value)
        coords_s=[]
        features_s=[]
        for patch_id in range(len(new_slide_info)):
            if new_slide_info[patch_id]['superpixel']==sup_value:
                coords_s.append(new_slide_info[patch_id]['coords'])
        coords_np=np.array(coords_s)
        patch_number=coords_np.shape[0]
        superpixel_patchnum[sup_value] = patch_number
    superpixels = modify_superpixels(superpixel_patchnum,superpixels)
    wsi_info = []
    for i in range(length):
        patch_info={}
        patch_info['patch_id']=i
        patch_info['coords']=f['coords'][i]
        patch_info['features']=f['features'][i]
        patch_coords=f['coords'][i]
        x,y=patch_coords 
        x=x+256
        y=y+256   
        x=int(x/16)   
        y=int(y/16)
        if (y<h)&(x<w):
            patch_info['superpixel']=superpixels[y][x]
        elif (y>=h)&(x<w):
            x_n,y_n=f['coords'][i]
            x_n=x_n+256
            y_n=y_n+(h*16-y_n)/2
            x_n=int(x_n/16)
            y_n=int(y_n/16)
            patch_info['superpixel']=superpixels[y_n][x_n]
        elif (y<h)&(x>=w):
            x_n,y_n=f['coords'][i]
            x_n=x_n+(w*16-x_n)/2
            y_n=y_n+256
            x_n=int(x_n/16)
            y_n=int(y_n/16)
            patch_info['superpixel']=superpixels[y_n][x_n]
        else:
            x_n,y_n=f['coords'][i]
            x_n=x_n+(w*16-x_n)/2
            y_n=y_n+(h*16-y_n)/2
            x_n=int(x_n/16)
            y_n=int(y_n/16)
            patch_info['superpixel']=superpixels[y_n][x_n]
        wsi_info.append(patch_info)
    torch.save(wsi_info,new_slide_info_path)
    return superpixels    

def generate_tissue_graph(slide_list_path,image_path,saved_path,graph_file_saved_path,vis_saved_path):
    """
    Generate a tissue graph for all the images in image path dir.
    """
    feature_extractor = DeepFeatureExtractor(
        architecture='resnet34',
        patch_size=144,
        resize_size=224
    )

    # 4. define graph builder
    tissue_graph_builder = RAGGraphBuilder(add_loc_feats=True)

    # 5. define graph visualizer
    visualizer = OverlayGraphVisualization()
#     print(fname)
        # b. extract superpixels
    slide_list = joblib.load(slide_list_path)
    # slide_x, _ = os.path.splitext(slide_list_path)
    _, slide_x = os.path.split(slide_list_path)
    # print(slide_list_path)
    # print(slide_x)
    slide_x = slide_x.split('_')[1]
    print(slide_x)
#     for slidename in os.listdir(image_path):
    for index in range(len(slide_list)):
        slidename = slide_list[index]
        name, _ = os.path.splitext(slidename)
#         print(name)
        stitch_name = name+'.jpg'
        stitch_path = os.path.join(image_path,stitch_name)
        print(stitch_path)
        if slide_x == '20x':
            superpixels = patch_judge_20x(stitch_path,saved_path)
        elif slide_x == '40x':
            superpixels = patch_judge_40x(stitch_path,saved_path)
#     torch.save(superpixels,os.path.join('/data13/yanhe/miccai/super_pixel/superpixel_array/tcga_lihc_200superpixel',image_name[:-4]+'.pt'))
#         print(image_name)
        print(superpixels.shape)
#     print(superpixels)
        # c. extract deep features
        image = Image.open(stitch_path)
        features = feature_extractor.process(image, superpixels)
#     print(features.shape)
        # d. build a Region Adjacency Graph (RAG)
#     graph = tissue_graph_builder.process(image, superpixels, features)
        graph = tissue_graph_builder.process(superpixels, features)
#     print(graph)
    # e. save the graph
        torch.save(graph,os.path.join(graph_file_saved_path,slidename[:-4]+'.pt'))
        # f. visualize and save the graph
        canvas = visualizer.process(image, graph, instance_map=superpixels)
        canvas.save(os.path.join(vis_saved_path,slidename[:-4]+'.png'))

def main(args):
    saved_path = args.saved_path
    graph_file_saved_path = args.graph_file_saved_path
    vis_saved_path = args.vis_saved_path
    os.makedirs(saved_path,exist_ok=True)
    os.makedirs(graph_file_saved_path,exist_ok=True)
    os.makedirs(vis_saved_path,exist_ok=True)
    #20x
    slide_20x_path = args.slide_20x_path
    stitch_20x_path = args.stitch_20x_path
    generate_tissue_graph(slide_20x_path,stitch_20x_path,saved_path,graph_file_saved_path,vis_saved_path)

    #40x
    slide_40x_path = args.slide_40x_path
    stitch_40x_path = args.stitch_40x_path
    generate_tissue_graph(slide_40x_path,stitch_40x_path,saved_path,graph_file_saved_path,vis_saved_path)



def get_params():
    parser = argparse.ArgumentParser(description='superpixel_generate')

    parser.add_argument('--slide_40x_path', type=str, default='/data12/yanhe/miccai/data/tcga_crc/slide_40x_list.pkl')
    parser.add_argument('--slide_20x_path', type=str, default='/data12/yanhe/miccai/data/tcga_crc/slide_20x_list.pkl')
    parser.add_argument('--stitch_20x_path', type=str, default='/data12/ybj/survival/CRC/20x/stitches')
    parser.add_argument('--stitch_40x_path', type=str, default='/data12/ybj/survival/CRC/40x/stitches')
    parser.add_argument('--saved_path', type=str, default='/data11/yanhe/miccai/super_pixel/slide_superpixel/tcga_crc/superpixel_num_300/')
    parser.add_argument('--vis_saved_path', type=str, default='/data11/yanhe/miccai/super_pixel/vis/tcga_crc/superpixel_num_300')
    parser.add_argument('--graph_file_saved_path', type=str, default='/data11/yanhe/miccai/super_pixel/graph_file/tcga_crc/superpixel_num_300')
    parser.add_argument('--nr_superpixels', type=int, default=300)  

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        args=get_params()
        main(args)
    except Exception as exception:
#         logger.exception(exception)
        raise
