## Multi-scope Analysis Driven Hierarchical Graph Transformer for Whole Slide Image based Cancer Survival Prediction

<div align=left><img width="70%" src="./images/miccai_framework.png"/></div>

## Installation
Clone the repo:
```bash
git clone https://github.com/Baeksweety/HGTHGT  && cd HGTHGT
```
Create a conda environment and activate it:
```bash
conda create -n env python=3.8
conda activate env
pip install -r requirements.txt
```

## Data Preprocess
***generate_superpixel.py*** shows how to generate merged superpixels of whole slide images and ***graph_construction.ipynb*** shows how to transform a  histological image into the hierarchical graphs. After the data processing is completed, put all hierarchical graphs into a folder. The form is as follows:
```bash
PYG_Data
   └── Dataset
          ├── pyg_data_1.pt
          ├── pyg_data_2.pt
                    :
          └── pyg_data_n.pt
```


## Cluster
***cluster.py*** shows how to generate the fixed number of clusters which woould be used in the train process. The form is as follows:
```bash
Cluster_Info
   └── Dataset
          ├── cluster_info_1.pt
          ├── cluster_info_2.pt
                    :
          └── cluster_info_n.pt
```



## Training
First, setting the data path, data splits and hyperparameters in the file ***train.py***. Then, experiments can be run using the following command-line:
```bash
cd train
python train.py
or
bash run.sh
```

## Saved models
We provide a 5-fold checkpoint for each dataset, which performing as:
| Dataset | CI |
| ----- |:--------:|
| CRC   | 0.607 |
| TCGA_LIHC | 0.657 |
| TCGA_KIRC | 0.646 |





## More Info
- Our implementation refers the following publicly available codes. 
  - [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric)--Fey M, Lenssen J E. Fast graph representation learning with PyTorch Geometric[J]. arXiv preprint arXiv:1903.02428, 2019.
  - [Histocartography](https://github.com/histocartography/histocartography)--Jaume G, Pati P, Anklin V, et al. HistoCartography: A toolkit for graph analytics in digital pathology[C]//MICCAI Workshop on Computational Pathology. PMLR, 2021: 117-128.
  - [ViT Pytorch](https://github.com/lukemelas/PyTorch-Pretrained-ViT)--Dosovitskiy A, Beyer L, Kolesnikov A, et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale[C]//International Conference on Learning Representations. 2020.
  - [NAGCN](https://github.com/YohnGuan/NAGCN)--Guan Y, Zhang J, Tian K, et al. Node-aligned graph convolutional network for whole-slide image representation and classification[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 18813-18823.
