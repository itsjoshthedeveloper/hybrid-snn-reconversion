#!/bin/bash
#SBATCH -p pi_panda --gres=gpu:1 --mem=20G -t 40:00:00 --mail-user=josh.chough@yale.edu --mail-type=ALL

module load miniconda
source activate pytorch_env
python snn_to_ann.py --gpu True -a VGG5 --pretrained_ann './trained_models/ann/ann_vgg5_cifar10.pth' --pretrained_snn './trained_models/snn_ans/08-29-2020_16-10-51_snn_vgg5_cifar10_100.pth' --log