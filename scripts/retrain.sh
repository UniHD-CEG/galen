#!/usr/bin/env bash

python -m tools.train --model resnet18_cifar --ckpt_load_path results/checkpoints/cifar/resnet18_cifar.pth --store_dir ./results/checkpoints/cifar/ \
      --batch_size 256 --train_mom 0.9 --weight_decay 5e-4 --train_lr 0.02 --epochs 30 \
       --wandb_entity uhdcsg --wandb_project retrain-torben --num_worker 1 \
       --log_dir ./logs/retrain \
       --policy ./logs/pq_r6_b-3.0_r0.25_cBOPs_-episode-100.pickle