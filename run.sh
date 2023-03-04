# #!/bin/bash

python interpretable_transformer.py  --seed 0 --fold_num 0 --label "tcga_lihc_fold0_superpixel200_cluster32_lr1e-5_30epoch"
python interpretable_transformer.py  --seed 0 --fold_num 1 --label "tcga_lihc_fold1_superpixel200_cluster32_lr1e-5_30epoch"
python interpretable_transformer.py  --seed 0 --fold_num 2 --label "tcga_lihc_fold2_superpixel200_cluster32_lr1e-5_30epoch"
python interpretable_transformer.py  --seed 0 --fold_num 3 --label "tcga_lihc_fold3_superpixel200_cluster32_lr1e-5_30epoch"
python interpretable_transformer.py  --seed 0 --fold_num 4 --label "tcga_lihc_fold4_superpixel200_cluster32_lr1e-5_30epoch"
# python graph_transformer.py --seed 3 --fold_num 0 --label "tcga_lihc_fold0_lr1e-5_30epoch" 
# python graph_transformer.py --seed 3 --fold_num 1 --label "tcga_lihc_fold1_lr1e-5_30epoch" 
# python graph_transformer.py --seed 3 --fold_num 2 --label "tcga_lihc_fold2_lr1e-5_30epoch" 
# python graph_transformer.py --seed 3 --fold_num 3 --label "tcga_lihc_fold3_lr1e-5_30epoch" 
# python graph_transformer.py --seed 3 --fold_num 4 --label "tcga_lihc_fold4_lr1e-5_30epoch" 
