#!/usr/bin/env bash

python -m tools.train --model resnet18_cifar --store_dir ./results/checkpoints/cifar/ \
      --batch_size 256 --train_mom 0.9 --weight_decay 5e-4 --train_lr 0.05  \
      --add_train_identifier pre-train --num_worker 2 --epochs 100