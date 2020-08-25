#!/bin/bash
#SBATCH -p pi_panda --gres=gpu:1 --mem=20G -t 40:00:00 --mail-user=josh.chough@yale.edu --mail-type=ALL

module load miniconda
source activate pytorch_env
python snn.py --epochs 100 --batch_size 32 --pretrained_ann './trained_models/ann/ann_vgg16_cifar10.pth' --log