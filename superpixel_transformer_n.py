from timm.models.vision_transformer import VisionTransformer
import timm.models.vision_transformer
import skimage.io as io
import argparse
import joblib
import copy
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import skimage.io as io
from timm.models.layers import drop_path, to_2tuple, trunc_normal_,PatchEmbed
from timm.models.helpers import build_model_with_cfg, named_apply
from torch_geometric.nn import global_mean_pool,global_max_pool,GlobalAttention,dense_diff_pool,global_add_pool,TopKPooling,ASAPooling,SAGPooling
from torch_geometric.nn import GCNConv,ChebConv,SAGEConv,GraphConv,LEConv,LayerNorm,GATConv
import torch
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
import torch.nn as nn
torch.set_num_threads(8)
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from functools import partial
from block_utils import Block
from torch_geometric.data import Data as geomData
from timm.models.layers import trunc_normal_
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

class my_GlobalAttention(torch.nn.Module):
    def __init__(self, gate_nn, nn=None):
        super(my_GlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)


    def forward(self, x, batch, size=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch.max().item() + 1 if size is None else size   #modified
        
        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, num_nodes=size)   
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        return out


class Intra_GCN(nn.Module):
    def __init__(self,in_feats,n_hidden,out_feats,drop_out_ratio=0.2,mpool_method="global_mean_pool",gnn_method='sage'):
        super(Intra_GCN,self).__init__() 
        if gnn_method == 'sage':       
            self.conv1= SAGEConv(in_channels=in_feats,out_channels=out_feats)  
        elif gnn_method == 'gcn':
            self.conv1= GCNConv(in_channels=in_feats,out_channels=out_feats)  
        elif gnn_method == 'gat':
            self.conv1= GATConv(in_channels=in_feats,out_channels=out_feats)  
        elif gnn_method == 'leconv':
            self.conv1= LEConv(in_channels=in_feats,out_channels=out_feats)  
        elif gnn_method == 'graphconv':
            self.conv1= GraphConv(in_channels=in_feats,out_channels=out_feats)  
        # self.conv2= SAGEConv(in_channels=n_hidden,out_channels=out_feats)
        
        self.relu = torch.nn.ReLU() 
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout=nn.Dropout(p=drop_out_ratio)
        self.softmax = nn.Softmax(dim=-1)
        
        if mpool_method == "global_mean_pool":
            self.mpool = global_mean_pool 
        elif mpool_method == "global_max_pool":
            self.mpool = global_max_pool 
        elif mpool_method == "global_att_pool":
            att_net=nn.Sequential(nn.Linear(out_feats, out_feats//2), nn.ReLU(), nn.Linear(out_feats//2, 1))     
            self.mpool = my_GlobalAttention(att_net)  
        self.norm = LayerNorm(in_feats)      
        self.norm2 = LayerNorm(out_feats)
        self.norm1 = LayerNorm(n_hidden)
        
    def forward(self,data):
        x=data.x
        edge_index = data.edge_patch

        x = self.norm(x)
        x = self.conv1(x,edge_index)
        x = self.relu(x)  
        # x = self.sigmoid(x)
        # x = self.norm(x)
        x = self.norm1(x)
        x = self.dropout(x)

        # x = self.conv2(x,edge_index)
        # x = self.relu(x)  
        # # x = self.sigmoid(x)
        # # x = self.norm(x)   
        # x = self.norm2(x)    
        # x = self.dropout(x)
        # print(x)

#         batch = torch.zeros(len(x),dtype=torch.long).to(device)
        batch = data.superpixel_attri.to(device)
        x = self.mpool(x,batch)
        # print('fea dim is {}'.format(x.shape))
        # print(x)
        
        fea = x
        # print(fea.shape)

        return fea

class Inter_GCN(nn.Module):
    def __init__(self,in_feats,n_hidden,out_feats,drop_out_ratio=0.2,mpool_method="global_mean_pool",gnn_method='sage'):
        super(Inter_GCN,self).__init__()        
        # self.conv1= SAGEConv(in_channels=in_feats,out_channels=out_feats)          
        # self.conv2= SAGEConv(in_channels=n_hidden,out_channels=out_feats)
        if gnn_method == 'sage':       
            self.conv1= SAGEConv(in_channels=in_feats,out_channels=out_feats)  
        elif gnn_method == 'gcn':
            self.conv1= GCNConv(in_channels=in_feats,out_channels=out_feats)  
        elif gnn_method == 'gat':
            self.conv1= GATConv(in_channels=in_feats,out_channels=out_feats)  
        elif gnn_method == 'leconv':
            self.conv1= LEConv(in_channels=in_feats,out_channels=out_feats)  
        elif gnn_method == 'graphconv':
            self.conv1= GraphConv(in_channels=in_feats,out_channels=out_feats)  
        
        self.relu = torch.nn.ReLU() 
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout=nn.Dropout(p=drop_out_ratio)
        self.softmax = nn.Softmax(dim=-1)
        
        if mpool_method == "global_mean_pool":
            self.mpool = global_mean_pool 
        elif mpool_method == "global_max_pool":
            self.mpool = global_max_pool 
        elif mpool_method == "global_att_pool":
            att_net=nn.Sequential(nn.Linear(out_feats, out_feats//2), nn.ReLU(), nn.Linear(out_feats//2, 1))     
            self.mpool = my_GlobalAttention(att_net)        
        self.norm = LayerNorm(in_feats)
        self.norm2 = LayerNorm(out_feats)
        self.norm1 = LayerNorm(n_hidden)
        
    def forward(self,data,feature):
        x=feature
        edge_index = data.edge_superpixel
        # print(x.shape)
        x = self.norm(x)
        x = self.conv1(x,edge_index)
        x = self.relu(x)  
        # x = self.sigmoid(x)
        x = self.norm1(x)
        x = self.dropout(x)

        # x = self.conv2(x,edge_index)
        # x = self.relu(x)  
        # # x = self.sigmoid(x)
        # x = self.norm2(x)       
        # x = self.dropout(x)

        # batch = torch.zeros(len(x),dtype=torch.long).to(device)
        # x = self.mpool(x,batch)
        
        fea = x
        # print(x.shape)
        # print(fea.shape)

        return fea



class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, num_patches=100,no_embed_class=False,class_token=True, depth=1,drop_path_rate=0.,mlp_ratio=4.,pre_norm=True,qkv_bias=True,init_values=None,drop_rate=0.,attn_drop_rate=0.,norm_layer=None,act_layer=None,weight_init='',global_pool='token', fc_norm=None,**kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        embed_dim = kwargs['embed_dim']
        self.patch_embed = nn.Linear(embed_dim,embed_dim)
        num_patches = num_patches
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.no_embed_class = no_embed_class
        self.global_pool = global_pool

        self.num_prefix_tokens = 1 if class_token else 0
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=kwargs['num_heads'],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)])
        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        # if self.grad_checkpointing and not torch.jit.is_scripting():
        #     x = checkpoint_seq(self.blocks, x)
        # else:
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]   #token
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return torch.sigmoid(x)

    def get_attention_weights(self):
        return [block.get_attention_weights() for block in self.blocks]

def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_moco(module: nn.Module, name: str = ''):
    """ ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def get_init_weights_vit(mode='jax', head_bias: float = 0.):
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm


class Superpixel_Vit(nn.Module):
    def __init__(self,in_feats_intra=1024,n_hidden_intra=1024,out_feats_intra=1024,in_feats_inter=1024,n_hidden_inter=1024,out_feats_inter=1024,vw_num=16,feat_dim=1024,num_classes=1,depth=1,num_heads = 16,final_fea_type = 'mean',mpool_intra='global_mean_pool',mpool_inter='global_mean_pool',gnn_intra='sage',gnn_inter='sage'):
        super(Superpixel_Vit, self).__init__()

        self.vw_num = vw_num
        self.feat_dim = feat_dim

        #intra-graph
        self.gcn1 = Intra_GCN(in_feats=in_feats_intra,n_hidden=n_hidden_intra,out_feats=out_feats_intra,mpool_method=mpool_intra,gnn_method=gnn_intra)

        #inter-graph
        self.gcn2 = Inter_GCN(in_feats=in_feats_inter,n_hidden=n_hidden_inter,out_feats=out_feats_inter,mpool_method=mpool_inter,gnn_method=gnn_inter)
        self.vit = VisionTransformer(num_patches = vw_num,num_classes = num_classes, embed_dim = feat_dim,depth = depth,num_heads = num_heads)

        self.final_fea_type = final_fea_type

    def superpixel_graph(self,data):
        superpixel_attri = data.superpixel_attri
        min_value = int(min(superpixel_attri))
        #intra-graph
        superpixel_fea = self.gcn1(data)
        #intra-graph
        superpixel_feas = self.gcn2(data,superpixel_fea)
        if min_value == 0:
            print('min superpixel value is 0')
            superpixel_fea_all={}
            for index in range(1,superpixel_feas.shape[0]):
                fea = superpixel_feas[index].unsqueeze(0)
                superpixel_value = index
                superpixel_fea_all[superpixel_value] = fea
        else:
            print('min superpixel value is 1')
            superpixel_fea_all={}
            for index in range(superpixel_feas.shape[0]):
                fea = superpixel_feas[index].unsqueeze(0)
                superpixel_value = index+1
                superpixel_fea_all[superpixel_value] = fea
        return superpixel_fea_all


    def mean_feature(self,superpixel_features,cluster_info):
        mask=np.zeros((self.vw_num,self.feat_dim))
        mask=torch.tensor(mask).to(device)
        # superpixel_cluster_path = os.path.join(cluster_info_path,slidename+'.pth')
        # cluster_info = torch.load(superpixel_cluster_path)
        for vw in range(self.vw_num):
            fea_all=torch.Tensor().to(device)
            for superpixel_value in cluster_info.keys(): 
                if cluster_info[superpixel_value]['cluster']==vw:
                    if fea_all.shape[0]==0:
                        fea_all=superpixel_features[superpixel_value]
                    else:
                        fea_all=torch.cat((fea_all,superpixel_features[superpixel_value]),dim=0)
            if fea_all.shape[0]!=0:
                fea_avg=torch.mean(fea_all,axis=0)
#             print('fea_avg shape:{}'.format(fea_avg.shape))
                mask[vw]=fea_avg
        return mask

    def max_feature(self,superpixel_features,cluster_info):
        mask=np.zeros((self.vw_num,self.feat_dim))
        mask=torch.tensor(mask).to(device)
        # superpixel_cluster_path = os.path.join(cluster_info_path,slidename+'.pth')
        # cluster_info = torch.load(superpixel_cluster_path)
        for vw in range(self.vw_num):
            fea_all=torch.Tensor().to(device)
            for superpixel_value in cluster_info.keys(): 
                if cluster_info[superpixel_value]['cluster']==vw:
                    if fea_all.shape[0]==0:
                        fea_all=superpixel_features[superpixel_value]
                    else:
                        fea_all=torch.cat((fea_all,superpixel_features[superpixel_value]),dim=0)
            if fea_all.shape[0]!=0:
                fea_max,_=torch.max(fea_all,dim=0)
#             print('fea_avg shape:{}'.format(fea_avg.shape))
                mask[vw]=fea_max
        return mask

    def forward(self,data,cluster_info):

        superpixels_fea = self.superpixel_graph(data)
        #final-fea
        if self.final_fea_type == 'mean':
            fea = self.mean_feature(superpixels_fea,cluster_info) #[16，1024]
        elif self.final_fea_type == 'max':
            fea = self.max_feature(superpixels_fea,cluster_info)
        fea = fea.unsqueeze(0)  #[1,16,1024]
        # print(fea.shape)
        # print(fea.shape)
        #vit
        fea = fea.float()
        out = self.vit(fea)
        return out

    def get_attention_weights(self):
        return self.vit.get_attention_weights()
