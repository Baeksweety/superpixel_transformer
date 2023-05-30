from timm.models.vision_transformer import VisionTransformer
import timm.models.vision_transformer
import skimage.io as io
import argparse
import joblib
import copy
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import skimage.io as io
from timm.models.layers import drop_path, to_2tuple, trunc_normal_,PatchEmbed
# from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv, checkpoint_seq
import torch
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
# from model_utils import Block,DropPath,get_sinusoid_encoding_table
import torch.nn as nn
torch.set_num_threads(8)
# from lifelines.utils import concordance_index
import numpy as np
from lifelines.utils import concordance_index as ci
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from functools import partial
from block_utils import Block
from superpixel_transformer_n import Superpixel_Vit

try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message


def _neg_partial_log(prediction, T, E):

    current_batch_len = len(prediction)
    R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_matrix_train[i, j] = T[j] >= T[i]

    train_R = torch.FloatTensor(R_matrix_train)
    train_R = train_R.cuda()

    # train_ystatus = torch.tensor(np.array(E),dtype=torch.float).to(device)
    train_ystatus =  E

    theta = prediction.reshape(-1)

    exp_theta = torch.exp(theta)
    loss_nn = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus)

    return loss_nn 

def get_val_ci(pre_time,patient_and_time,patient_sur_type):
    ordered_time, ordered_pred_time, ordered_observed=[],[],[]
    for x in pre_time:
        ordered_time.append(patient_and_time[x])
        ordered_pred_time.append(pre_time[x]*-1)
        ordered_observed.append(patient_sur_type[x])
#     print(len(ordered_time), len(ordered_pred_time), len(ordered_observed))
    return ci(ordered_time, ordered_pred_time, ordered_observed)


def get_train_val(train_slide,val_slide,test_slide,label_path):
    train_slidename=[]
    train_label_censorship=[]
    train_label_survtime=[]
    val_slidename=[]
    val_label_censorship=[]
    val_label_survtime=[]
    test_slidename=[]
    test_label_censorship=[]
    test_label_survtime=[]
    label = torch.load(label_path)

    for index in train_slide:
        train_slidename.append(index)
        train_label_censorship.append(label[index]['censorship'])   
        train_label_survtime.append(label[index]['surv_time'])

    for index1 in val_slide:
        val_slidename.append(index1)
        val_label_censorship.append(label[index1]['censorship'])
        val_label_survtime.append(label[index1]['surv_time'])

    for index2 in test_slide:
        test_slidename.append(index2)
        test_label_censorship.append(label[index2]['censorship'])
        test_label_survtime.append(label[index2]['surv_time'])

    return train_slidename,val_slidename,test_slidename,train_label_censorship,train_label_survtime,val_label_censorship,val_label_survtime,test_label_censorship,test_label_survtime



def eval_model(model,test_slide,test_label_surv_type,test_label_time,batch_size,cuda=True):
    model = model.cuda()
    model = model.to(device)
    model = model.eval()

    #print('begin evaluation!')
    with torch.no_grad():
        running_loss=0.
        test_loss = 0.
        t_out_pre = torch.Tensor().to(device)
        t_labelall_surv_type = torch.Tensor().to(device)
        t_labelall_time = torch.Tensor().to(device)

        atten_dict = {}
        total_loss_for_test = 0
        # batch_id = 0 
        test_num = len(test_slide)
        # batch divide
        iter_num =  test_num //  batch_size +  1

        # batch
        for batch_iter in range(iter_num):
            outs = torch.Tensor().to(device)
            labels_surv_type = torch.Tensor().to(device)
            labels_time = torch.Tensor().to(device)
            
            if (batch_iter + 1) == iter_num:  # last batch
                for sample_index in range(batch_iter * batch_size, test_num):
#                     print("Sample Index: ", sample_index)
                    
                    slidename = test_slide[sample_index]
                    pyg_data = torch.load(os.path.join(args.pyg_path,slidename+'.pt')).to(device)
                    print(pyg_data)
                    label_surv_type = torch.tensor(test_label_surv_type[sample_index]).unsqueeze(0).to(device)
                    label_time = torch.tensor(test_label_time[sample_index]).unsqueeze(0).to(device)
                    
                    cluster_info = torch.load(os.path.join(args.cluster_info_path,slidename+'.pth'))
                    # model train
                    output = model(pyg_data,cluster_info)
                    atten = model.get_attention_weights()
                    average_atten = torch.mean(atten[0],dim=1)   #record attention score
                    atten_dict[label_time[0]] = average_atten[0]
                    # save output and label to calculate loss
                    if outs.shape[0] == 0:
                        outs = output
                    else:
                        outs = torch.cat((outs, output), dim=0)
                    
                    if labels_surv_type.shape[0] == 0:
                        labels_surv_type = label_surv_type
                    else:
                        labels_surv_type = torch.cat((labels_surv_type, label_surv_type), dim=0)
                    
                    if labels_time.shape[0] == 0:
                        labels_time = label_time
                    else:
                        labels_time = torch.cat((labels_time, label_time),dim=0)
                    
                    # save all samples results
                    
                    if t_out_pre.shape[0] == 0:
                        t_out_pre = -1 * output
                    else:
                        t_out_pre = torch.cat((t_out_pre, -1 * output),dim=0)
                    
                    if t_labelall_surv_type.shape[0] == 0:
                        t_labelall_surv_type = label_surv_type
                    else:
                        t_labelall_surv_type = torch.cat((t_labelall_surv_type, label_surv_type), dim=0)
                        
                    if t_labelall_time.shape[0] == 0:
                        t_labelall_time = label_time
                    else:
                        t_labelall_time = torch.cat((t_labelall_time, label_time), dim=0)
                    # calculate loss of the batch
                if torch.sum(labels_surv_type) > 0.0:
                    print("outs.shape:",outs.shape,"labels_time.shape:",labels_time.shape)                  
                    loss = _neg_partial_log(outs,labels_time,labels_surv_type)
                    loss = args.cox_loss * loss

                    total_loss_for_test += loss.item()
                    print("Batch Avg Loss: {:.4f}", loss.item())
                    
            else:
                # batch
                for batch_index in range(batch_size):
                    index = batch_iter * batch_size + batch_index
#                     print("Sample Index: ", index)
                    
                    # acquire dat and label
                    slidename = test_slide[index]
                    pyg_data = torch.load(os.path.join(args.pyg_path,slidename+'.pt')).to(device)
                    print(pyg_data)
                    label_surv_type = torch.tensor(test_label_surv_type[index]).unsqueeze(0).to(device)
                    label_time = torch.tensor(test_label_time[index]).unsqueeze(0).to(device)
                    
                    cluster_info = torch.load(os.path.join(args.cluster_info_path,slidename+'.pth'))
                    # model train
                    output = model(pyg_data,cluster_info)
                    atten = model.get_attention_weights()
                    average_atten = torch.mean(atten[0],dim=1)   
                    atten_dict[label_time[0]] = average_atten[0]
                    # save the output and label
                    if outs.shape[0] == 0:
                        outs = output
                    else:
                        outs = torch.cat((outs, output), dim=0)
                    
                    if labels_surv_type.shape[0] == 0:
                        labels_surv_type = label_surv_type
                    else:
                        labels_surv_type = torch.cat((labels_surv_type, label_surv_type), dim=0)
                    
                    if labels_time.shape[0] == 0:
                        labels_time = label_time
                    else:
                        labels_time = torch.cat((labels_time, label_time),dim=0)
                    
                    # save all samples results
                    
                    if t_out_pre.shape[0] == 0:
                        t_out_pre = -1 * output
                    else:
                        t_out_pre = torch.cat((t_out_pre, -1 * output),dim=0)
                    
                    if t_labelall_surv_type.shape[0] == 0:
                        t_labelall_surv_type = label_surv_type
                    else:
                        t_labelall_surv_type = torch.cat((t_labelall_surv_type, label_surv_type), dim=0)
                        
                    if t_labelall_time.shape[0] == 0:
                        t_labelall_time = label_time
                    else:
                        t_labelall_time = torch.cat((t_labelall_time, label_time), dim=0)
                    #compute the loss
                if torch.sum(labels_surv_type) > 0.0:
                    print("outs.shape:",outs.shape,"labels_time.shape:",labels_time.shape)

                    
                    loss = _neg_partial_log(outs,labels_time,labels_surv_type)
                    loss = args.cox_loss * loss

                    total_loss_for_test += loss.item()
                    print("Batch Avg Loss: {:.4f}", loss.item())

        c_idx_epochs_avg = ci(t_labelall_time.data.cpu(),t_out_pre.data.cpu(),t_labelall_surv_type.data.cpu())
        epoch_loss_test = total_loss_for_test / iter_num
        print("epoch_val_test_loss:{}".format(epoch_loss_test))

        return c_idx_epochs_avg,epoch_loss_test,atten_dict


def train_model(n_epochs,model,optimizer,scheduler,train_slide,val_slide,test_slide,train_label_censorship,train_label_survtime,val_label_censorship,val_label_survtime,test_label_censorship,test_label_survtime,fold_num,batch_size,cuda=True):    
    if cuda:
        model = model.to(device)
    
    os.makedirs("/data14/yanhe/miccai/train/saved_model/tcga_kirc/interpreatable_transformer_depth_"+str(args.depth)+"/seed_"+str(args.seed)+"/",exist_ok=True)
    os.makedirs("/data14/yanhe/miccai/train/log_result/tcga_kirc/interpreatable_transformer_depth_"+str(args.depth)+"/seed_"+str(args.seed),exist_ok=True)
    os.makedirs("/data14/yanhe/miccai/train/attention_weights/tcga_kirc/interpreatable_transformer_depth_"+str(args.depth)+"/seed_"+str(args.seed)+'/atten_dict/',exist_ok=True)

    best_loss = 1e9
    best_ci = 0.
    best_acc = 0.
    n_out_features = 1
    n_classes = [1]*n_out_features
    for epoch in range(n_epochs):
        model.train()   
#                 pbar.set_description("RepeatNum:{}/{} Seed:{} Fold:{}/{}".format(repeat_num_temp + 1,repeat_num,seed,fold_num,all_fold_num))
        total_loss_for_train = 0
        batch_id = 0
        out_pre = torch.Tensor().to(device)
        pre_for_batch = torch.Tensor().to(device)
        labelall_surv_type = torch.Tensor().to(device)
        label_surv_type_for_batch = torch.Tensor().to(device)
        label_time_for_batch = torch.Tensor().to(device)
        # print(labelall_surv_type.shape[0])
        labelall_time = torch.Tensor().to(device)
        train_num = len(train_slide)
        print("train_slide Num: ",len(train_slide))
        
        
        iter_num =  train_num //  batch_size +  1
                
        for batch_iter in range(iter_num):
            outs = torch.Tensor().to(device)
            labels_surv_type = torch.Tensor().to(device)
            labels_time = torch.Tensor().to(device)
            
            if (batch_iter + 1) == iter_num:  
                for sample_index in range(batch_iter * batch_size, train_num):
#                     print("Sample Index: ", sample_index)
                    
                    slidename = train_slide[sample_index]
                    pyg_data = torch.load(os.path.join(args.pyg_path,slidename+'.pt')).to(device)
                    print(pyg_data)
                    label_surv_type = torch.tensor(train_label_censorship[sample_index]).unsqueeze(0).to(device)
                    label_time = torch.tensor(train_label_survtime[sample_index]).unsqueeze(0).to(device)
                    
                    cluster_info = torch.load(os.path.join(args.cluster_info_path,slidename+'.pth'))

                    output = model(pyg_data,cluster_info)

                    if outs.shape[0] == 0:
                        outs = output
                    else:
                        outs = torch.cat((outs, output), dim=0)
                    
                    if labels_surv_type.shape[0] == 0:
                        labels_surv_type = label_surv_type
                    else:
                        labels_surv_type = torch.cat((labels_surv_type, label_surv_type), dim=0)
                    
                    if labels_time.shape[0] == 0:
                        labels_time = label_time
                    else:
                        labels_time = torch.cat((labels_time, label_time),dim=0)
                    
                    
                    if out_pre.shape[0] == 0:
                        out_pre = -1 * output
                    else:
                        out_pre = torch.cat((out_pre, -1 * output),dim=0)
                    
                    if labelall_surv_type.shape[0] == 0:
                        labelall_surv_type = label_surv_type
                    else:
                        labelall_surv_type = torch.cat((labelall_surv_type, label_surv_type), dim=0)
                        
                    if labelall_time.shape[0] == 0:
                        labelall_time = label_time
                    else:
                        labelall_time = torch.cat((labelall_time, label_time), dim=0)

                if torch.sum(labels_surv_type) > 0.0:
                    # print("outs.shape:",outs.shape,"labels_time.shape:",labels_time.shape)

                    
                    loss = _neg_partial_log(outs,labels_time,labels_surv_type)
                    loss = args.cox_loss * loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss_for_train += loss.item()
                    # print("Batch Avg Loss: {:.4f}", loss.item())
                    
            else:

                for batch_index in range(batch_size):
                    index = batch_iter * batch_size + batch_index
#                     print("Sample Index: ", index)

                    slidename = train_slide[index]
                    pyg_data = torch.load(os.path.join(args.pyg_path,slidename+'.pt')).to(device)
                    print(pyg_data)
                    label_surv_type = torch.tensor(train_label_censorship[index]).unsqueeze(0).to(device)
                    label_time = torch.tensor(train_label_survtime[index]).unsqueeze(0).to(device)
                    
                    cluster_info = torch.load(os.path.join(args.cluster_info_path,slidename+'.pth'))

                    output = model(pyg_data,cluster_info)
                    
                    if outs.shape[0] == 0:
                        outs = output
                    else:
                        outs = torch.cat((outs, output), dim=0)
                    
                    if labels_surv_type.shape[0] == 0:
                        labels_surv_type = label_surv_type
                    else:
                        labels_surv_type = torch.cat((labels_surv_type, label_surv_type), dim=0)
                    
                    if labels_time.shape[0] == 0:
                        labels_time = label_time
                    else:
                        labels_time = torch.cat((labels_time, label_time),dim=0)
                    
                    
                    if out_pre.shape[0] == 0:
                        out_pre = -1 * output
                    else:
                        out_pre = torch.cat((out_pre, -1 * output),dim=0)
                    
                    if labelall_surv_type.shape[0] == 0:
                        labelall_surv_type = label_surv_type
                    else:
                        labelall_surv_type = torch.cat((labelall_surv_type, label_surv_type), dim=0)
                        
                    if labelall_time.shape[0] == 0:
                        labelall_time = label_time
                    else:
                        labelall_time = torch.cat((labelall_time, label_time), dim=0)

                if torch.sum(labels_surv_type) > 0.0:
                    # print("outs.shape:",outs.shape,"labels_time.shape:",labels_time.shape)

                    
                    loss = _neg_partial_log(outs,labels_time,labels_surv_type)
                    loss = args.cox_loss * loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss_for_train += loss.item()
                    # print("Batch Avg Loss: {:.4f}", loss.item())
                    
                    
                    
        c_idx_for_train = ci(labelall_time.data.cpu(),out_pre.data.cpu(),labelall_surv_type.data.cpu())
        
        epoch_loss = total_loss_for_train / iter_num
        
        print("Epoch [{}/{}], epoch_loss {:.4f}".format(epoch+1,n_epochs, epoch_loss))                    

        #val
        val_c_idx,val_loss,_ = eval_model(model,val_slide,val_label_censorship,val_label_survtime,batch_size,cuda=cuda)

        #test
        test_c_idx,test_loss,_ =eval_model(model,test_slide,test_label_censorship,test_label_survtime,batch_size,cuda=cuda)
        # if scheduler is not None:
        #     scheduler.step()
       
        with open('/data14/yanhe/miccai/train/log_result/tcga_kirc/interpreatable_transformer_depth_'+str(args.depth)+"/seed_"+str(args.seed)+'/'+args.label+'_random.log',"a") as f:
            f.write(f"EPOCH {epoch} : \n")
            f.write(f"train loss - {epoch_loss} train ci - {c_idx_for_train};\n")
            f.write(f"val loss - {val_loss} val ci -{val_c_idx};\n")
            f.write(f"test loss - {test_loss}test ci - {test_c_idx};\n")
        if val_c_idx >= best_ci:
            best_epoch = epoch
            best_ci = val_c_idx
            t_model = copy.deepcopy(model)
        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     best_epoch = epoch
        #     t_model = copy.deepcopy(model)
    # t_model = copy.deepcopy(model)
    save_path = '/data14/yanhe/miccai/train/saved_model/tcga_kirc/interpreatable_transformer_depth_'+str(args.depth)+"/seed_"+str(args.seed)+'/fold_num_{}.pth'.format(fold_num)
    torch.save({'model_state_dict': t_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, save_path)
    print("Model saved: %s" % save_path)

    t_test_c_idx,t_test_loss,atten_dict = eval_model(t_model,test_slide,test_label_censorship,test_label_survtime,batch_size,cuda=cuda)

    atten_dict_saved_path = '/data14/yanhe/miccai/train/attention_weights/tcga_kirc/interpreatable_transformer_depth_'+str(args.depth)+"/seed_"+str(args.seed)+'/atten_dict/'+args.label+'_foldnum_{}.pth'.format(fold_num)
    torch.save(atten_dict,atten_dict_saved_path)
    with open('/data14/yanhe/miccai/train/log_result/tcga_kirc/interpreatable_transformer_depth_'+str(args.depth)+"/seed_"+str(args.seed)+'/'+args.label+'_random.log',"a") as f:
        f.write(f"best model test ci value {t_test_c_idx} occurs at EPOCH {best_epoch} ;\n")
        # f.write(f"best model test ci value {t_test_c_idx} ;\n")


def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def main(args):
    fold_num = args.fold_num
    label_path = args.label_path
    n_epochs = args.epochs
    batch_size = args.batch_size
    label = torch.load(label_path)
    split_dict_path = args.split_dict_path
    seed = args.seed
    pyg_path = args.pyg_path
    print("seed:{}".format(seed))
    #splits
    # split_dict=torch.load('/data14/yanhe/miccai/data/tcga_lihc/train_val_test_split.pkl')
    split_dict = joblib.load(split_dict_path)
    fold_num = args.fold_num
    fold_train = 'fold_'+str(fold_num)+'_train'
    fold_val = 'fold_'+str(fold_num)+'_val'
    fold_test = 'fold_'+str(fold_num)+'_test'
    train_slide = split_dict[fold_train]  
    val_slide = split_dict[fold_val]
    test_slide = split_dict[fold_test]
    setup_seed(seed)

    model = Superpixel_Vit(in_feats_intra=args.in_feats_intra,
        n_hidden_intra=args.n_hidden_intra,
        out_feats_intra=args.out_feats_intra,
        in_feats_inter=args.in_feats_inter,
        n_hidden_inter=args.n_hidden_inter,
        out_feats_inter=args.out_feats_inter,
        vw_num=args.vw_num,
        feat_dim=args.feat_dim,
        num_classes=1,
        depth=args.depth,
        num_heads = args.num_heads,
        final_fea_type = args.final_fea_type,
        mpool_intra=args.mpool_intra,
        mpool_inter=args.mpool_inter,
        gnn_intra=args.gnn_intra,
        gnn_inter=args.gnn_inter
    )
    
    optimizer = torch.optim.AdamW([dict(params=model.parameters(), lr=args.lr, betas=(0.9, 0.95),weight_decay = args.l2_reg_alpha),])
    t_max = n_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=t_max, eta_min=0. )


    Train_slide,Val_slide,Test_slide,Train_Labels_censorship,Train_Labels_survtime,Val_Labels_censorship,Val_Labels_survtime,Test_Labels_censorship,Test_Labels_survtime = get_train_val(train_slide,val_slide,test_slide,label_path)
    train_model(n_epochs, model, optimizer, scheduler, Train_slide,Val_slide,Test_slide,Train_Labels_censorship,Train_Labels_survtime,Val_Labels_censorship,Val_Labels_survtime,Test_Labels_censorship,Test_Labels_survtime,fold_num,batch_size,cuda=True)





def get_params():
    parser = argparse.ArgumentParser(description='model training')

    parser.add_argument('--label_path',type=str, default='/data12/yanhe/miccai/data/tcga_kirc/slide_label.pt')
    parser.add_argument('--split_dict_path',type=str, default='/data12/yanhe/miccai/data/tcga_kirc/train_val_test_split_random1.pkl')
    parser.add_argument('--pyg_path',type=str, default='/data14/yanhe/miccai/graph_file/tcga_kirc/superpixel_num_600')
    parser.add_argument('--cluster_info_path',type=str, default='/data14/yanhe/miccai/codebook/cluster_info/tcga_kirc/superpixel600_cluster16/all_fold' )
    parser.add_argument('--vw_num',type=int, default=16)
    parser.add_argument('--feat_dim',type =int,default=1024)
    parser.add_argument('--depth',type=int,default=1)   
    parser.add_argument('--num_heads',type=int,default=4)
    parser.add_argument('--mpool_intra',type=str,default='global_mean_pool') #‘global_mean_pool’,'global_max_pool','global_att_pool'
    parser.add_argument('--mpool_inter',type=str,default='global_mean_pool')
    parser.add_argument('--gnn_intra',type=str,default='sage') #'sage''gcn''gat''leconv''graphconv'
    parser.add_argument('--gnn_inter',type=str,default='sage')
    parser.add_argument('--in_feats_intra',type=int, default=1024)
    parser.add_argument('--n_hidden_intra',type=int, default=1024)
    parser.add_argument('--out_feats_intra',type=int,default=1024)
    parser.add_argument('--in_feats_inter',type=int,default=1024)   #in_feats_inter=out_feats_intra
    parser.add_argument('--n_hidden_inter',type=int, default=1024)
    parser.add_argument('--out_feats_inter',type=int,default=1024)  #out_feats_inter=feat_dim
    parser.add_argument('--final_fea_type',type=str,default='mean')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size',type=int,default=16)
    #parser.add_argument('--warmup_epochs', type=int, default= 40)
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate of model training")
    parser.add_argument("--l2_reg_alpha",type=float,default=0.001)
    parser.add_argument("--cox_loss",type=float,default=12)
    parser.add_argument("--seed",type=int, default=1)
    parser.add_argument("--fold_num",type=int, default=0)
    parser.add_argument("--label",type=str,default="tcga_lihc_fold_0_lr1e-5_30epoch")

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

