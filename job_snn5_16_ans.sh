#!/bin/bash
#SBATCH -p pi_panda --gres=gpu:1 --mem=20G -t 40:00:00 --mail-user=josh.chough@yale.edu --mail-type=ALL

module load miniconda
source activate pytorch_env
python snn_ans.py -a VGG5 --epochs 16 --lr_reduce 5 --pretrained_ann './trained_models/snn_ans/ann_vgg5_cifar10.pth' --log