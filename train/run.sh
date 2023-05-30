# #!/bin/bash

python train.py  --mpool_intra 'global_max_pool' --seed 1 --fold_num 0 --label "tcga_argo_sage_max_1024featdim_fold0_superpixel600_cluster16_numhead8_lr1e-5_30epoch_l2regalpha0.001_batchsize_16_split0_depth1"
python train.py  --mpool_intra 'global_max_pool' --seed 1 --fold_num 1 --label "tcga_argo_sage_max_1024featdim_fold1_superpixel600_cluster16_numhead8_lr1e-5_30epoch_l2regalpha0.001_batchsize_16_split0_depth1"
python train.py  --mpool_intra 'global_max_pool' --seed 1 --fold_num 2 --label "tcga_argo_sage_max_1024featdim_fold2_superpixel600_cluster16_numhead8_lr1e-5_30epoch_l2regalpha0.001_batchsize_16_split0_depth1"
python train.py  --mpool_intra 'global_max_pool' --seed 1 --fold_num 3 --label "tcga_argo_sage_max_1024featdim_fold3_superpixel600_cluster16_numhead8_lr1e-5_30epoch_l2regalpha0.001_batchsize_16_split0_depth1"
python train.py  --mpool_intra 'global_max_pool' --seed 1 --fold_num 4 --label "tcga_argo_sage_max_1024featdim_fold4_superpixel600_cluster16_numhead8_lr1e-5_30epoch_l2regalpha0.001_batchsize_16_split0_depth1"

#according to the dataset and parameters, the value of label should be changed
