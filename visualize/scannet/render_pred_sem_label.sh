#!/bin/bash

data_dir="/local-scratch/qiruiw/research/dense-scanrefer/log/scannet/pointgroup/test/2021-02-10_01-53-53/split_pred/val/semantic"
stk_dir="/local-scratch/qiruiw/research/stk-motifs"
script="$stk_dir/ssc/render-file.js"
config_dir="$stk_dir/ssc/config"
config_file="/project/3dlg-hcvc/dense-scanrefer/www/scannet/render_turntable.json"
output_dir="/project/3dlg-hcvc/dense-scanrefer/www/scannet/sem_seg/pred_sem_label"
scene_ids="/local-scratch/qiruiw/research/dense-scanrefer/data/scannet/meta_data/scannetv2_val.txt"
n=16
​
parallel -j $n --eta "CUDA_VISIBLE_DEVICES=0 $script \
    --input $data_dir/{1}.ply \
    --output_dir $output_dir/$split/{1} \
    --config_file $config_file \
    --assetType=model >& $output_dir/log/{1}.render.log" :::: $scene_ids
